"""Measure a demanded service on the native scheduler, reusing qualified structure when possible."""

from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import read_json
from prologue.host.acquisition.recording import SessionCapture
from prologue.host.acquisition.rules import EvidenceGap
from prologue.host.performance.requirements import CostRequirements, MeasuredWork
from prologue.host.runtime.docker import DockerImage
from prologue.integrations.bindings import RuleName

from ..contracts import LocalCut, LocalRule, WorkerDomain
from ..evidence import CostSamples
from ..host_samples import HostCostSamples
from ..readback import WorkerCapture
from ..recipes.scheduler import LocalSchedulerBatch, LocalWorkerBinding


class LocalSchedulerWork(LocalRule):
    compiler_gap = "missing_work_compiler"

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query.get("invocation")

    @staticmethod
    def answers(queries, report, asset, model=None, directory=None):
        missing = MeasuredWork(report).uncovered(queries, model, directory)
        answers = [{"status": "untriggered", "detail": missing[i]} if i in missing
                   else {"status": "qualified", "asset": asset} for i in range(len(queries))]
        return LocalRule.asset(asset, report, answers)

    @staticmethod
    def native_invocation(query):
        return "invocation" in query and LocalSchedulerBatch.supports(query["invocation"] | {"domain": query["domain"]})


class LocalDeviceWork(LocalSchedulerWork):
    name = RuleName("scheduler_device_service")

    def matches(self, query):
        return query.get("work") in {"device", "transfer"} and self.native_invocation(query)

    def cut(self, queries, session, deployment, arguments):
        return LocalSchedulerBatch.work_job(queries, deployment)

    def qualify(self, queries, recording, timeout=None):
        worker, = WorkerCapture.observed(recording)
        report, _ = CostSamples.worker_services(worker, self.compiler(timeout), recording.directory.parent)
        return self.answers(queries, report, "samples/device.json")


class LocalPeerWork(LocalSchedulerWork):
    name = RuleName("peer_service")

    def matches(self, query):
        operand = query.get("features", {}).get("transport")
        return (query.get("work") == "collective" and operand and len(operand["members"]) > 1
                and self.native_invocation(query))

    @staticmethod
    def session_key(query, deployment):
        return LocalSchedulerWork.session_key(query, deployment), query["features"]["transport"]["members"]

    def cut(self, queries, session, deployment, arguments):
        cut = LocalSchedulerBatch.work_job(queries, deployment)
        peers = session[1]
        if cut.job["single_device"] or any(q["features"]["transport"]["members"] != peers for q in queries):
            raise EvidenceGap("missing_physical_peers", required_ranks=peers, physical_devices=cut.cards)

        # Service after rendezvous requires simultaneous arrivals from every peer.
        cut.job["batch"]["peer_ranks"] = peers
        return cut

    def qualify(self, queries, recording, timeout=None):
        model = self.compiler(timeout)
        with TemporaryDirectory(dir=recording.directory.parent) as scratch:
            directory = Path(scratch)
            report = CostSamples.transport(recording, queries, model, directory)

            return self.answers(queries, report, "samples/transport.json", model, directory)


class LocalHostWork(LocalSchedulerWork):
    name = RuleName("scheduler_host_service")

    priority = 10

    selected_catalog = None

    def matches(self, query):
        return query.get("work") == "host" and WorkerDomain.native_domain(query["domain"])

    def sources(self, deployment, rank):
        for directory in self.arguments.structure_session:
            receipt = read_json(directory / "result.json")
            if (receipt.get("rule") not in {LocalSchedulerBatch.name, LocalWorkerBinding.name}
                    or receipt["status"] not in {"qualified", "incomplete"}):
                continue
            original = SessionCapture(directory, receipt)
            if original.inputs["deployment"] == deployment:
                yield from (worker for worker in WorkerCapture.observed(original) if worker.rank == rank)

    def cut(self, queries, session, deployment, arguments):
        domain = queries[0]["domain"]
        requirements = CostRequirements()
        needed = {requirements.gap("query", q) for q in queries}
        image = DockerImage.id(arguments.source_image)
        model = self.compiler(arguments.preparation_seconds)

        for worker in self.sources(deployment, WorkerDomain.rank(domain)):
            source = worker.recording
            job = source.inputs["job"]
            if (source.inputs["image"] != image
                    or WorkerDomain.execution_domain(worker.read("facts")["environment"]) != domain):
                continue

            _, catalog = worker.measure(model, arguments.output.parent)
            points = {point for _, point in requirements.observe_graph("source", catalog)}
            if needed <= points:
                self.selected_catalog = source.attempt_id, worker.rank, catalog
                return LocalCut(job, source.inputs["physical_devices"], source.inputs["framework_revision"],
                                job["model"]["path"], {"template_recording": source.attempt_id},
                                worker="--local-host", host_timing=True)

        raise EvidenceGap("missing_local_structure_source", rank=WorkerDomain.rank(domain))

    def qualify(self, queries, recording, timeout=None):
        inputs = recording.inputs
        rank = WorkerDomain.rank(queries[0]["domain"])
        original = next((w for w in self.sources(inputs["deployment"], rank)
                         if w.recording.attempt_id == inputs["template_recording"]), None)

        if original is None:
            raise EvidenceGap("missing_local_structure_source", required_recording=inputs["template_recording"])
        timed = WorkerCapture(recording, rank)
        if self.selected_catalog and self.selected_catalog[:2] == (original.recording.attempt_id, rank):
            catalog = self.selected_catalog[2]
        else:
            _, catalog = original.measure(self.compiler(timeout), recording.directory.parent)

        report = HostCostSamples.local(original, timed, catalog)
        return self.answers(queries, report, "samples/host.json")


rules = (LocalHostWork, LocalDeviceWork, LocalPeerWork)
