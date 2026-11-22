"""Publish native SGLang/Ascend work samples from a recorded production."""

import json
from collections import Counter
from pathlib import Path

from prologue.acquisition.observation.workers import CapturedWorkers
from prologue.acquisition.recording import PublishedCapture
from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.acquisition.pd_samples import PdNotificationSamples
from prologue.integrations.platforms.ascend.collectives import CollectiveServices
from prologue.integrations.platforms.ascend.services import ObservedServices
from prologue.integrations.platforms.ascend.transport_profile import AscendTransportProfile
from prologue.performance.samples import ServiceSamples, WorkCatalog


class CostSamples:
    def __init__(self, production, projection, ranks):
        self.production = production.resolve()
        root = projection or production
        inputs = json.loads((root / "production.json").read_text())["inputs"]
        self.recording = PublishedCapture.from_source(Path(inputs["capture_path"]))
        if root != production:
            original = json.loads((production / "production.json").read_text())["inputs"]
            if inputs["capture_path"] != original["capture_path"]:
                raise ValueError("regenerated cost evidence belongs to a different native observation")
        observed = json.loads((root / "result.json").read_text())["observed_roles"]
        workers = CapturedWorkers(inputs["capture"]["instances"])
        if observed != workers.observed or set(ranks) - set(observed):
            raise ValueError("cost samples require an observed production rank")

        self.work = root / "work"
        self.joint = workers.joint_execution
        self.facts = {}
        self.collectives = {}
        self.notifications = None
        self.peer_measurements = False
        if not self.joint:
            return

        self.facts = {rank: self.read_facts(rank) for rank in ranks}
        measured = [body["measurements"] for body in self.facts.values()]
        collective = any(row["collectives"] for row in measured)
        pd_protocol = any("pd_protocol" in row for row in measured)

        # A PD notification or a collective needs its peers from the same physical execution.
        self.peer_measurements = collective or pd_protocol
        if self.peer_measurements:
            self.facts.update((rank, self.read_facts(rank)) for rank in observed if rank not in self.facts)
        if collective:
            self.collectives = CollectiveServices(self.facts, self.recording.source, ranks).samples
        if pd_protocol:
            protocol = {rank: body["measurements"]["pd_protocol"] for rank, body in self.facts.items()}
            self.notifications = PdNotificationSamples(protocol)

    def read_facts(self, rank):
        return json.loads((self.work / f"{rank}-facts.json").read_text())

    @classmethod
    def export(cls, arguments):
        jobs = tuple(zip(arguments.rank, arguments.work_identities, arguments.output, strict=True))
        ranks = tuple(arguments.rank)
        if len(set(ranks)) != len(jobs) or len({path.resolve() for _, _, path in jobs}) != len(jobs):
            raise ValueError("cost sample exports require distinct ranks and output paths")

        source = cls(arguments.production, arguments.projection, ranks)
        for rank, identities, path in jobs:
            catalog = WorkCatalog(source.work / f"{rank}-graph.json", identities)
            report = source.measure(rank, catalog, transport_only=arguments.transport_only)
            write_json(path, report, replace=False)

    def measure(self, rank, catalog, *, transport_only=False):
        observation = {"production": str(self.production), "rank": rank}
        facts = self.facts[rank] if self.joint else self.read_facts(rank)
        services = ObservedServices(catalog.graph, facts, self.collectives.get(rank, {}), physical_peers=self.joint)
        recordings = [self.recording.attempt_id]
        if transport_only:
            if not self.peer_measurements:
                raise ValueError("a transport profile requires a joint physical observation of the peer group")
            profile = AscendTransportProfile(self.recording.source, self.facts, self.work / f"{rank}-sdk/trace.json",
                                             observation, self.recording.inputs["image"])
            return {**profile.samples(catalog, services), "native_recordings": recordings}

        samples, absent = ServiceSamples(catalog.domain, catalog.definitions), Counter()
        for piece, definition, interval, features in catalog:
            duration_ns, copy, collective = services.service(piece)
            if duration_ns is None:
                absent[piece["kind"]] += 1
                continue
            if "copy" in piece:
                if copy != [features["axes"]["transfer.bytes"], piece["copy"]["direction"]]:
                    raise ValueError("native copy size or direction differs from its bound tensor operand")

                # The structural SDK times the serving call. The ready-operand model trains on its own timing window.
                absent["requires_ready_operand_timing"] += 1
                continue

            # Source request and physical resource IDs locate the observation only.
            samples.add(definition, interval, features, duration_ns, copy, collective)
        notifications = list(self.notifications.intervals(rank)) if self.notifications is not None else []
        for measured in notifications:
            span = measured["duration_bounds_ns"]
            if span is None:
                absent["notification"] += 1
                continue
            features = {"phase": "", "axes": {"parameter.dependencies": measured["dependencies"]}}
            samples.add_external({"operation": measured["operation"]}, "transfer", features, sum(span) // 2)

        report = samples.document(absent)
        report["observation"] = observation
        report["native_recordings"] = recordings
        if notifications:
            report["notification_measurements"] = notifications
        return report
