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
    def __init__(self, arguments):
        self.output = arguments.output
        production, rank = arguments.production, arguments.rank
        self.observation = {"production": str(production.resolve()), "rank": rank}
        root = arguments.projection or production
        inputs = json.loads((root / "production.json").read_text())["inputs"]
        recording = PublishedCapture.from_source(Path(inputs["capture_path"]))
        self.attempt_id = recording.attempt_id
        if arguments.projection is not None:
            original = json.loads((production / "production.json").read_text())["inputs"]
            if inputs["capture_path"] != original["capture_path"]:
                raise ValueError("regenerated cost evidence belongs to a different native observation")
        ranks = json.loads((root / "result.json").read_text())["observed_roles"]
        workers = CapturedWorkers(inputs["capture"]["instances"])
        if ranks != workers.observed or rank not in ranks:
            raise ValueError("cost samples require an observed production rank")
        work = root / "work"
        self.catalog = WorkCatalog(work / f"{rank}-graph.json", arguments.work_identities)
        facts = json.loads((work / f"{rank}-facts.json").read_text())
        measurement = facts["measurements"]
        peers = {}
        joint = workers.joint_execution

        # A PD notification or a collective needs its peers from the same physical execution.
        if joint and (measurement["collectives"] or "pd_protocol" in measurement):
            peers = {actor: facts if actor == rank else json.loads((work / f"{actor}-facts.json").read_text())
                     for actor in ranks}
        collectives = {}
        if peers and measurement["collectives"]:
            collectives = CollectiveServices(peers, Path(inputs["capture_path"]), rank).samples
        self.services = ObservedServices(self.catalog.graph, facts, collectives, physical_peers=joint)
        self.profile = None
        if arguments.transport_only:
            if not joint or not peers:
                raise ValueError("a transport profile requires a joint physical observation of the peer group")
            self.profile = AscendTransportProfile(Path(inputs["capture_path"]), peers, work / f"{rank}-sdk/trace.json",
                                                  self.observation, recording.inputs["image"])
        self.notifications = []
        if peers and "pd_protocol" in measurement:
            protocol = {actor: body["measurements"]["pd_protocol"] for actor, body in peers.items()}
            self.notifications = list(PdNotificationSamples(protocol).intervals(rank))

    def run(self):
        if self.profile is not None:
            profile = self.profile.samples(self.catalog, self.services)
            write_json(self.output, {**profile, "native_recordings": [self.attempt_id]}, replace=False)
            return
        samples, absent = ServiceSamples(self.catalog.domain, self.catalog.definitions), Counter()
        for piece, definition, interval, features in self.catalog:
            duration_ns, copy, collective = self.services.service(piece)
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
        for measured in self.notifications:
            span = measured["duration_bounds_ns"]
            if span is None:
                absent["notification"] += 1
                continue
            features = {"phase": "", "axes": {"parameter.dependencies": measured["dependencies"]}}
            samples.add_external({"operation": measured["operation"]}, "transfer", features, sum(span) // 2)

        report = samples.document(absent)
        report["observation"] = self.observation
        report["native_recordings"] = [self.attempt_id]
        if self.notifications:
            report["notification_measurements"] = self.notifications
        write_json(self.output, report, replace=False)
