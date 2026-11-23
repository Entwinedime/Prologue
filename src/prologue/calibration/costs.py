"""Fit exact-coordinate medians from separate native device and host observations."""

import json
from collections import defaultdict
from statistics import median

from prologue.acquisition.observation.batches import ObservedBatches
from prologue.integrations.platforms.ascend.evidence import AscendEvidence


class Calibration:
    def __init__(self):
        self.domain = None
        self.batches = []
        self.observations = []
        self.sources = []

    def device(self, directory):
        observation = AscendEvidence.open(directory)
        evidence = observation.document()
        domain = {**evidence["domain"], "hardware": observation.hardware(directory)}
        if self.domain is not None and self.domain != domain:
            raise ValueError("device observations are from different cost domains")
        self.domain = domain
        self.record_source(observation.manifest)
        for batch in ObservedBatches(evidence).entries:
            self.batches.append({"action": batch.action(), "graph": batch.graph})
            samples = []
            for row in batch.measurements():
                samples.append({key: row[key] for key in ("duration_ns", "dispatch_ns", "submission_ns")})
            self.observations.append((None, samples))

    def host(self, directory):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["mode"] != "host-timing":
            raise ValueError("host fitting requires thread-time observations")
        if any(self.domain[key] != manifest[key] for key in ("framework_revision", "model", "configuration")):
            raise ValueError("host observations are from a different native configuration")
        if manifest["initial_state"] != {"kv": "empty", "workspace": "warmed"}:
            raise ValueError("host observations require the warmed empty-KV initial state")
        self.record_source(manifest)
        worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        for row in worker["host_work"]:
            members = row["members"]
            self.batches.append({"action": {"phase": row["phase"], "members": members},
                                 "graph": {"calls": [], "work": []}})
            sample = {
                "prepare_ns": row["prepare_cpu_ns"], "forward_ns": row["forward_cpu_ns"],
                "result_ns": row["result_cpu_ns"],
                "publications_ns": [row["publication_cpu_ns"][member["request"]] for member in members],
            }
            self.observations.append((sample, []))

    def record_source(self, manifest):
        self.sources.append({"mode": manifest["mode"], "origin_raw_ns": manifest["delivery"]["origin_raw_ns"]})

    def fit(self, coordinates):
        samples = defaultdict(list)
        for coordinate, (host, device) in zip(coordinates, self.observations, strict=True):
            if host is not None:
                samples[json.dumps(coordinate["host"], sort_keys=True)].append(host)
            for work, record in zip(coordinate["device"], device, strict=True):
                samples[json.dumps(work, sort_keys=True)].append(record)

        models = []
        for coordinate, rows in sorted(samples.items()):
            estimate = {}
            for key, first in rows[0].items():
                if isinstance(first, list):
                    estimate[key] = [round(median(row[key][index] for row in rows)) for index in range(len(first))]
                else:
                    estimate[key] = round(median(row[key] for row in rows))
            models.append({"coordinate": json.loads(coordinate), "estimate": estimate, "samples": len(rows)})
        return {"domain": self.domain, "estimator": "exact_median", "sources": self.sources, "models": models}
