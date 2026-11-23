"""Fit exact-coordinate medians from separate native device and host observations."""

import json
from collections import Counter, defaultdict
from statistics import median

from prologue.acquisition.observation.batches import ObservedBatches, aligned_batches, aligned_collectives
from prologue.integrations.platforms.ascend.collectives import PhysicalCollectives
from prologue.integrations.platforms.ascend.evidence import AscendEvidence, AscendRecording


class Calibration:
    def __init__(self, rules, query=None):
        self.rules = rules
        self.query = query
        self.domain = None
        self.batches = []
        self.observations = []
        self.sources = []

    def device(self, directory):
        recording = AscendRecording(directory, modes=("detailed", "decode-attention"))
        family = "device_attention" if recording.mode == "decode-attention" else "device_task"
        self.record_source(family, recording.mode, recording.delivery)
        ranks = [None] if recording.windows is None else sorted(recording.profiles)
        for rank in ranks:
            profile = directory if rank is None else recording.profile_directory(rank)
            self.device_profile(recording.read(rank), AscendEvidence.hardware(profile))

    def device_domain(self, domain, hardware):
        expected = {**domain, "hardware": hardware}
        if self.domain is not None:
            actual_hardware = {key: value for key, value in self.domain["hardware"].items() if key != "transport_links"}
            if {**self.domain, "hardware": actual_hardware} != expected:
                raise ValueError("device observations are from different cost domains")
        else:
            self.domain = expected

    def device_profile(self, observation, hardware):
        evidence = observation.document()
        self.device_domain(evidence["domain"], hardware)
        for batch in ObservedBatches(evidence).entries:
            communication = {scope for scope, _ in batch.collectives()}
            work, recorded = [], []
            for node, row in zip(batch.graph["work"], batch.device, strict=True):
                if row["owner"] not in communication:
                    work.append(node)
                    recorded.append(row)
            self.batches.append({"action": batch.action(), "graph": {**batch.graph, "work": work}})
            samples = []
            for row in batch.measurements(recorded):
                samples.append({key: row[key] for key in ("duration_ns", "dispatch_ns", "submission_ns")})
            self.observations.append((None, samples))

    def collective(self, directory):
        recording = AscendRecording(directory)
        physical = PhysicalCollectives(recording)
        self.device_domain(recording.domain, physical.hardware)
        links = self.domain["hardware"].setdefault("transport_links", physical.transport_links)
        if links != physical.transport_links:
            raise ValueError("the collective samples use a different physical topology")

        measured = 0
        unavailable = Counter()
        ranks = [ObservedBatches(rank.evidence).entries for rank in physical.ranks]
        for batches in aligned_batches(ranks):
            graph = {"calls": [], "work": [], "collectives": []}
            samples = []
            for entries, transfers in aligned_collectives(batches):
                parameters = entries[0][1]["parameters"]
                gates = [rank.gates.get(scope) for rank, (scope, _) in zip(physical.ranks, entries, strict=True)]
                if None in gates:
                    unavailable["missing_stream_gate"] += len(entries)
                    continue
                ready = max(gate.ready_raw_ns for gate in gates)
                if any(gate.completed_raw_ns < ready for gate in gates):
                    unavailable["completion_before_group_readiness"] += len(entries)
                    continue

                group_index = len(graph["collectives"])
                group = {"kind": parameters["kind"]}
                if transfers is not None:
                    group["transfers"] = transfers
                graph["collectives"].append(group)
                for rank, ((_, call), gate) in enumerate(zip(entries, gates, strict=True)):
                    duration = gate.completed_raw_ns - ready
                    path = [rank, *call["path"]]
                    graph["calls"].append({**call, "path": path})
                    graph["work"].append({"path": path, "slot": 0, "ordinal": 0, "collective": group_index,
                                          "kind": "collective", "name": parameters["kind"]})

                    # Service starts at device readiness. Host submission is outside this coordinate.
                    samples.append({"duration_ns": duration, "dispatch_ns": 0, "submission_ns": 0})
                measured += len(entries)
            self.batches.append({"action": batches[0].action(), "graph": graph})
            self.observations.append((None, samples))

        self.record_source("collective", recording.mode, recording.delivery)
        self.sources[-1].update(measured_results=measured, unattested_results=sum(unavailable.values()))
        if unavailable:
            self.sources[-1]["unattested_reasons"] = dict(unavailable)

    def host(self, directory):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if any(self.domain[key] != manifest[key] for key in ("framework_revision", "model", "configuration")):
            raise ValueError("host observations are from a different native configuration")
        if manifest["initial_state"] != {"kv": "empty", "workspace": "warmed"}:
            raise ValueError("host observations require the warmed empty-KV initial state")
        self.record_source("host_thread", manifest["mode"], manifest["delivery"])
        worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        for row in worker["host_work"]:
            members = row["members"]
            self.batches.append({"action": {"phase": row["phase"], "members": members},
                                 "graph": {"calls": [], "work": []}})
            sample = {
                "prepare_ns": row["prepare_cpu_ns"], "forward_ns": row["forward_cpu_ns"],
                "result_ns": row["result_cpu_ns"],
                "publications_ns": [row["publication_cpu_ns"][member["request"]] for member in members
                                    if member.get("publishes", True)],
            }
            self.observations.append((sample, []))

    def record_source(self, family, mode, delivery):
        self.rules.require_observation(family, mode)
        self.sources.append({"family": family, "mode": mode, "origin_raw_ns": delivery["origin_raw_ns"]})

    def fit(self, coordinates):
        wanted = None
        if self.query is not None:
            if self.query["domain"] != self.domain:
                raise ValueError("the sampled physical domain does not match the cost queries")
            wanted = {json.dumps(row, sort_keys=True) for row in self.query["coordinates"]}

        samples = defaultdict(list)
        for coordinate, (host, device) in zip(coordinates, self.observations, strict=True):
            if host is not None:
                samples[json.dumps(coordinate["host"], sort_keys=True)].append(host)
            for work, record in zip(coordinate["device"], device, strict=True):
                samples[json.dumps(work, sort_keys=True)].append(record)

        models = []
        for identity, rows in sorted(samples.items()):
            if wanted is not None and identity not in wanted:
                continue
            coordinate = json.loads(identity)
            estimator = self.rules.estimator(coordinate["family"])
            if estimator != "exact_median":
                raise ValueError(f"no fitter is registered for {estimator}")
            estimate = {}
            for key, first in rows[0].items():
                if isinstance(first, list):
                    estimate[key] = [round(median(row[key][index] for row in rows)) for index in range(len(first))]
                else:
                    estimate[key] = round(median(row[key] for row in rows))
            models.append({"coordinate": coordinate, "estimator": estimator, "estimate": estimate,
                           "samples": len(rows)})
        return {"domain": self.domain, "sources": self.sources, "models": models}
