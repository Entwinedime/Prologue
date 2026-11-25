"""Extract measured services from native production and local batch recordings."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import ServiceSamples, WorkCatalog, cost_features
from prologue.common.io import write_json
from prologue.common.observation.workers import CapturedWorkers
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.performance.fitting import FittedCosts
from prologue.host.performance.requirements import CostRequirements
from prologue.integrations.bindings import platform_backend
from prologue.integrations.frameworks.sglang.acquisition.batch_readback import BatchReadback
from prologue.integrations.frameworks.sglang.acquisition.local_copy import LocalCopies
from prologue.integrations.frameworks.sglang.acquisition.local_embedding import LocalEmbedding
from prologue.integrations.frameworks.sglang.acquisition.pd_samples import PdNotificationSamples
from prologue.integrations.frameworks.sglang.acquisition.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain


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
        self.collectives = None
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
            self.collectives = platform_backend().collectives(self.facts, self.worker_sources(), ranks)
        if pd_protocol:
            protocol = {rank: body["measurements"]["pd_protocol"] for rank, body in self.facts.items()}
            self.notifications = PdNotificationSamples(protocol)

    def read_facts(self, rank):
        return json.loads((self.work / f"{rank}-facts.json").read_text())

    def worker_sources(self):
        return dict.fromkeys(self.facts, self.recording.source)

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

    @staticmethod
    def structural_worker(recording):
        paired = "probes" in recording.manifest
        if not paired and recording.inputs["rule"] != BatchReadback.rule:
            return None

        # The structural half of a paired CPU measurement already observed the real device work at this shape.
        rank = recording.inputs["job"]["batch"].get("rank", 0)
        return WorkerCapture(recording, rank, "structure" if paired else None)

    @classmethod
    def product(cls, recording, query):
        rank = WorkerDomain.rank(query["domain"])
        if query["work"] in {"device", "transfer"}:
            if LocalCopies.matches(query):
                return None
            worker = cls.structural_worker(recording)
            return "derived/device-services" if worker is not None and worker.rank == rank else None
        rule = recording.inputs["rule"]
        observed = {int(actor) for actor in recording.manifest.get("workers", {"0": "."})}
        peers = query["features"].get("transport", {}).get("members", ())
        physical = recording.inputs["deployment"].get("acquisition_mode", "full") == "full"
        if (
            query["work"] == "collective"
            and rank in observed
            and len(peers) > 1
            and physical
            and observed.issuperset(peers)
            and (rule == LocalEmbedding.name or "peer_ranks" in recording.inputs["job"].get("batch", {}))
        ):
            return f"derived/peer-service-{rank}"
        return None

    @classmethod
    def extract(cls, session, recording, queries, model, workspace):
        query = queries[0]
        if query["work"] != "collective":
            return cls.batch(session, model, workspace)

        with TemporaryDirectory(dir=workspace) as scratch:
            table = cls.transport(recording, queries, model, Path(scratch))
        return table | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def batch(cls, session, model, workspace):
        recording = PublishedCapture(session / "native")
        receipt = json.loads((session / "result.json").read_bytes())
        if (receipt["status"] not in {"qualified", "incomplete"} or receipt["native_recording"] != recording.attempt_id
                or receipt["rule"] != recording.inputs["rule"] or receipt["queries"] != recording.inputs["queries"]):
            raise ValueError("local costs require the qualified session owning this native recording")

        worker = cls.structural_worker(recording)
        if worker is None:
            raise ValueError("local costs require a structural worker observation")
        services, asset = cls.worker_services(worker, model, workspace)
        accepted = {answer["asset"] for answer in receipt["answers"] if answer["status"] == "qualified"}
        if not accepted or ("probes" not in recording.manifest and accepted != {asset}):
            raise ValueError("the qualified session and its observed native batch boundary disagree")
        return services | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def worker_services(cls, worker, model, workspace):

        # Project work and timing from the same recording. Local staging cannot measure service between peers.
        with TemporaryDirectory(dir=workspace) as scratch:
            facts, catalog = worker.catalog(model, Path(scratch))
            asset = BatchReadback.observed_asset(worker.job["batch"], worker.read("boundary"), facts)
            return cls.worker_table(facts, catalog), asset

    @classmethod
    def worker_table(cls, facts, catalog):
        services = platform_backend().observed_services(catalog.graph, facts, {}, physical_peers=False)
        samples, absent = cls.local_samples(catalog, services)
        return samples.document(absent)

    @staticmethod
    def uncovered(queries, report, model, directory):
        demand = CostRequirements()
        points = [demand.gap("query", q) for q in queries]
        _, missing, _ = demand.check(model, FittedCosts({"physical": report}, []).domains, directory)
        return {i: missing[p] for i, p in enumerate(points) if p in missing}

    @classmethod
    def transport(cls, recording, queries, model, directory):
        if recording.inputs["rule"] == LocalEmbedding.name:
            ranks = [int(key) for key in recording.manifest["workers"]]
            sources = dict.fromkeys(ranks, recording.source)
            traces = {actor: recording.source / f"sdk-{actor}/trace.json" for actor in ranks}
        else:
            sources = {worker.rank: worker.source for worker in WorkerCapture.observed(recording)}
            traces = {actor: source / "sdk/trace.json" for actor, source in sources.items()}
        rank = WorkerDomain.rank(queries[0]["domain"])
        if any(
            WorkerDomain.rank(q["domain"]) != rank or not sources.keys() >= set(q["features"]["transport"]["members"])
            for q in queries
        ):
            raise ValueError("transport service requires one worker and an observed physical peer group")
        return cls.peer_service(recording, sources, traces, rank, model, directory)

    @staticmethod
    def peer_service(recording, sources, traces, rank, model, directory):
        if recording.inputs["deployment"].get("acquisition_mode", "full") != "full":
            raise ValueError("transport calibration requires the full physical deployment")
        peers = {actor: WorkerCapture.observed_facts(source, actor, traces[actor]) for actor, source in sources.items()}
        local = peers[rank]
        catalog = WorkerCapture.project_work(local, model, directory)
        clocks = platform_backend().collectives(peers, sources, [rank])
        services = platform_backend().observed_services(catalog.graph, local, clocks.samples[rank], physical_peers=True)
        origin = {"native_recording": recording.attempt_id, "rank": rank}
        profile = platform_backend().transport_profile(sources, peers, traces[rank], origin,
                                         recording.inputs["image"], recording.inputs["deployment"])
        samples, absent = profile.samples(catalog, services)
        report = samples.document(absent) | {"transport_profile": profile.qualification}
        if conflicts := clocks.notification_conflicts[rank]:
            report["collective_notification_conflicts"] = conflicts
        return report

    def measure(self, rank, catalog, *, transport_only=False):
        observation = {"production": str(self.production), "rank": rank}
        facts = self.facts[rank] if self.joint else self.read_facts(rank)
        collective_services = self.collectives.samples[rank] if self.collectives is not None else {}
        services = platform_backend().observed_services(
            catalog.graph, facts, collective_services, physical_peers=self.joint
        )
        notifications = list(self.notifications.intervals(rank)) if self.notifications is not None else []
        if transport_only:
            if not self.peer_measurements:
                raise ValueError("a transport profile requires a joint physical observation of the peer group")
            profile = platform_backend().transport_profile(
                self.worker_sources(),
                self.facts,
                self.work / f"{rank}-sdk/trace.json",
                observation,
                self.recording.inputs["image"],
                self.recording.inputs["deployment"],
            )
            if notifications and "state_transfer_protocol" not in profile.qualification:
                raise ValueError("physical notification samples require their measured transport protocol")
            samples, absent = profile.samples(catalog, services)
            provenance = {"transport_profile": profile.qualification}
        else:
            samples, absent = self.local_samples(catalog, services)
            provenance = {"observation": observation}

        for measured in notifications:
            span = measured["duration_bounds_ns"]
            if span is None:
                absent["notification"] += 1
                continue
            features = cost_features({"parameter.dependencies": measured["dependencies"]})
            samples.add_external({"operation": measured["operation"]}, "transfer", features, sum(span) // 2)

        report = {**samples.document(absent), **provenance, "native_recordings": [self.recording.attempt_id]}
        if notifications:
            report["notification_measurements"] = notifications
        if self.collectives is not None and (conflicts := self.collectives.notification_conflicts[rank]):
            report["collective_notification_conflicts"] = conflicts
        return report

    @staticmethod
    def local_samples(catalog, services):
        samples, absent = ServiceSamples(catalog.domain, catalog.definitions), Counter()
        for piece, definition, interval, features in catalog:
            duration_ns, copy, collective = services.service(piece)
            if duration_ns is None:
                absent[piece["kind"]] += 1
                continue
            if collective is not None and "transport" not in features:
                absent["undeclared_peer_operand"] += 1
                continue
            if "copy" in piece:
                if copy != [features["axes"]["transfer.bytes"], piece["copy"]["direction"]]:
                    raise ValueError("native copy size or direction differs from its bound tensor operand")

                # The structural SDK times the serving call. The ready-operand model trains on its own timing window.
                absent["requires_ready_operand_timing"] += 1
                continue

            # Source request and physical resource IDs locate the observation only.
            samples.add(definition, interval, features, duration_ns, copy, collective)
        return samples, absent
