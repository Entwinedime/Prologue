"""Extract device and transport services with their original physical observation coverage."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import ServiceSamples, WorkCatalog, cost_features
from prologue.common.io import read_json, write_json
from prologue.common.observation import CapturedWorkers
from prologue.host.acquisition.recording import PublishedCapture, SessionCapture
from prologue.integrations.bindings import platform_backend

from .contracts import CopyOperand, WorkerDomain, embedding_rule, scheduler_batch_rule
from .host_calls import PdNotificationSamples
from .readback import BatchReadback, WorkerCapture


class CostSamples:
    batch_rule = scheduler_batch_rule

    def __init__(self, production, projection, ranks):
        self.production = production.resolve()
        root = projection or production
        inputs = read_json(root / "production.json")["inputs"]
        self.recording = PublishedCapture.from_source(Path(inputs["capture_path"]))

        if (root != production and inputs["capture_path"]
                != read_json(production / "production.json")["inputs"]["capture_path"]):
            raise ValueError("regenerated cost evidence belongs to a different native observation")
        observed = read_json(root / "result.json")["observed_roles"]
        workers = CapturedWorkers(inputs["capture"]["instances"])

        if observed != workers.observed or set(ranks) - set(observed):
            raise ValueError("cost samples require an observed production rank")

        self.work = root / "work"
        self.joint = workers.joint_execution
        self.facts = {rank: self.read_facts(rank) for rank in ranks} if self.joint else {}
        self.collectives = self.notifications = None

        if self.joint:
            measured = [body["measurements"] for body in self.facts.values()]
            collective = any(row["collectives"] for row in measured)
            pd = any("pd_protocol" in row for row in measured)

            # A measurement between workers needs its peers from the same execution.
            if collective or pd:
                self.facts.update((rank, self.read_facts(rank)) for rank in observed if rank not in self.facts)
            self.sources = dict.fromkeys(self.facts, self.recording.source)
            if collective:
                self.collectives = platform_backend().collectives(self.facts, self.sources, ranks)

            if pd:
                peers = {rank: body["measurements"]["pd_protocol"] for rank, body in self.facts.items()}
                self.notifications = PdNotificationSamples(peers)

    def read_facts(self, rank):
        return read_json(self.work / f"{rank}-facts.json")

    @classmethod
    def export(cls, arguments):
        jobs = tuple(zip(arguments.rank, arguments.work_identities, arguments.output, strict=True))
        ranks = arguments.rank
        if len(set(ranks)) != len(jobs) or len({path.resolve() for _, _, path in jobs}) != len(jobs):
            raise ValueError("cost sample exports require distinct ranks and output paths")

        source = cls(arguments.production, arguments.projection, ranks)
        for rank, identities, path in jobs:
            catalog = WorkCatalog(source.work / f"{rank}-graph.json", identities)
            report = source.measure(rank, catalog, transport_only=arguments.transport_only)
            write_json(path, report, replace=False)

    @classmethod
    def structural_rank(cls, recording):

        # A paired CPU probe also contains one independently observed device cut at this shape.
        if "probes" in recording.manifest or recording.inputs["rule"] == cls.batch_rule:
            return recording.inputs["job"]["batch"].get("rank", 0)
        return None

    @classmethod
    def product(cls, recording, query):
        rank = WorkerDomain.rank(query["domain"])
        if query["work"] in {"device", "transfer"}:
            if cls.structural_rank(recording) == rank and not CopyOperand.matches(query):
                return "derived/device-services"
            return None

        rule = recording.inputs["rule"]
        observed = {int(actor) for actor in recording.manifest.get("workers", {"0": "."})}
        peers = query["features"].get("transport", {}).get("members", ())
        physical = recording.inputs["deployment"].get("acquisition_mode", "full") == "full"
        peer_run = rule == embedding_rule or "peer_ranks" in recording.inputs["job"].get("batch", {})

        if (query["work"] == "collective" and physical and peer_run and rank in observed
                and len(peers) > 1 and observed.issuperset(peers)):
            return f"derived/peer-service-{rank}"
        return None

    @classmethod
    def extract(cls, session, recording, queries, model, workspace):
        if queries[0]["work"] != "collective":
            return cls.batch(session, model, workspace, recording=recording)

        with TemporaryDirectory(dir=workspace) as scratch:
            report = cls.transport(recording, queries, model, Path(scratch))
        return report | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def batch(cls, session, model, workspace, *, recording=None):
        recording = recording or SessionCapture(session)
        receipt = recording.receipt

        rank = cls.structural_rank(recording)
        if rank is None:
            raise ValueError("local costs require a structural worker observation")
        worker = WorkerCapture(recording, rank, "structure" if "probes" in recording.manifest else None)
        services, asset = cls.worker_services(worker, model, workspace)
        accepted = {answer["asset"] for answer in receipt["answers"] if answer["status"] == "qualified"}

        if not accepted or ("probes" not in recording.manifest and accepted != {asset}):
            raise ValueError("the qualified session and its observed native batch boundary disagree")
        return services | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def worker_services(cls, worker, model, workspace):

        # Timing and the graph have one source. A staged rank cannot measure service between peers.
        facts, catalog = worker.measure(model, workspace)
        asset = BatchReadback.observed_asset(worker.job["batch"], worker.read("boundary"), facts)
        return cls.worker_table(facts, catalog), asset

    @classmethod
    def worker_table(cls, facts, catalog):
        services = platform_backend().observed_services(catalog.graph, facts, {}, physical_peers=False)
        samples, absent = cls.local_samples(catalog, services)
        return samples.document(absent)

    @classmethod
    def transport(cls, recording, queries, model, directory):
        embedded = recording.inputs["rule"] == embedding_rule
        sources = (dict.fromkeys(map(int, recording.manifest["workers"]), recording.source) if embedded
                   else {worker.rank: worker.source for worker in WorkerCapture.observed(recording)})
        traces = {r: path / (f"sdk-{r}" if embedded else "sdk") / "trace.json" for r, path in sources.items()}
        rank = WorkerDomain.rank(queries[0]["domain"])

        if any(WorkerDomain.rank(q["domain"]) != rank or set(q["features"]["transport"]["members"]) - sources.keys()
               for q in queries):
            raise ValueError("transport service requires one worker and an observed physical peer group")

        if recording.inputs["deployment"].get("acquisition_mode", "full") != "full":
            raise ValueError("transport calibration requires the full physical deployment")
        peers = {actor: WorkerCapture.observed_facts(source, actor, traces[actor]) for actor, source in sources.items()}
        catalog = WorkerCapture.project_work(peers[rank], model, directory)
        backend = platform_backend()
        clocks = backend.collectives(peers, sources, [rank])
        services = backend.observed_services(catalog.graph, peers[rank], clocks.samples[rank], physical_peers=True)
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
            if self.collectives is None and self.notifications is None:
                raise ValueError("a transport profile requires a joint physical observation of the peer group")
            profile = platform_backend().transport_profile(self.sources, self.facts,
                        self.work / f"{rank}-sdk/trace.json", observation,
                        self.recording.inputs["image"], self.recording.inputs["deployment"])

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
            duration, copy, collective = services.service(piece)
            if duration is None:
                absent[piece["kind"]] += 1
            elif collective is not None and "transport" not in features:
                absent["undeclared_peer_operand"] += 1
            elif "copy" in piece:
                if copy != [features["axes"]["transfer.bytes"], piece["copy"]["direction"]]:
                    raise ValueError("native copy size or direction differs from its bound tensor operand")

                # Operand-ready timing comes from the isolated copy probe.
                absent["requires_ready_operand_timing"] += 1
            else:
                samples.add(definition, interval, features, duration, copy, collective)

        return samples, absent
