"""Resolve service queries against work actually measured in a native scheduler cut."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.recording import PublishedCapture
from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples
from prologue.integrations.frameworks.sglang.ascend.host_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.ascend.local_attention import LocalAttention
from prologue.integrations.frameworks.sglang.ascend.local_copy import LocalCopies
from prologue.integrations.frameworks.sglang.ascend.local_scheduler import LocalSchedulerBatch, LocalWorkerBinding
from prologue.integrations.frameworks.sglang.ascend.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.runtimes.docker import ModelingProcess
from prologue.performance.requirements import CostRequirements


class LocalSchedulerWork:
    def __init__(self, arguments):
        self.arguments = arguments

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query.get("invocation")

    def model(self, seconds):
        if self.arguments.binary is None:
            raise EvidenceGap("missing_work_compiler", required="--binary")
        timeout = self.arguments.timeout if seconds is None else min(self.arguments.timeout, seconds)
        return ModelingProcess(self.arguments.binary, self.arguments.modeling_image, timeout)

    @staticmethod
    def answers(queries, report, asset):
        demand = CostRequirements()
        measured = set(demand.samples("probe", report, basis="native_measurement"))
        answers = []
        for query in queries:
            if demand.gap("query", query) in measured:
                answers.append({"status": "qualified", "asset": asset})
            else:
                answers.append({"status": "untriggered", "detail": "requested_work_interval_was_not_measured"})
        return ({asset: report} if any(a["status"] == "qualified" for a in answers) else {}), answers


class LocalDeviceWork(LocalSchedulerWork):
    name = "sglang.ascend.scheduler_device_service"

    @staticmethod
    def matches(query):
        return (query.get("work") in {"device", "transfer"} and "invocation" in query
                and not LocalAttention.matches(query) and not LocalCopies.matches(query)
                and LocalSchedulerBatch.supports(query["invocation"] | {"domain": query["domain"]}))

    @staticmethod
    def prepare(queries, deployment, arguments, limit):
        job, devices, revision = LocalSchedulerBatch.work_job(queries, deployment)
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, devices, revision, job,
                                                 "--local-worker", runtime_trace=True, model=job["model"]["path"])

    def qualify(self, queries, recording, timeout=None):
        worker, = WorkerCapture.observed(recording)
        report, _ = CostSamples.worker_services(worker, self.model(timeout), recording.directory.parent)
        return self.answers(queries, report, "samples/device.json")


class LocalPeerWork(LocalSchedulerWork):
    name = "sglang.ascend.peer_service"

    @staticmethod
    def matches(query):
        operand = query.get("features", {}).get("transport")
        return (query.get("work") == "collective" and operand and len(operand["members"]) > 1
                and "invocation" in query
                and LocalSchedulerBatch.supports(query["invocation"] | {"domain": query["domain"]}))

    @staticmethod
    def session_key(query, deployment):
        return LocalSchedulerWork.session_key(query, deployment), query["features"]["transport"]["members"]

    @staticmethod
    def prepare(queries, deployment, arguments, limit):
        job, devices, revision = LocalSchedulerBatch.work_job(queries, deployment)
        peers = queries[0]["features"]["transport"]["members"]
        if job["single_device"] or any(q["features"]["transport"]["members"] != peers for q in queries):
            raise EvidenceGap("missing_physical_peers", required_ranks=peers, physical_devices=devices)

        # A duration after rendezvous uses a simultaneously observed arrival from every required peer.
        job["batch"]["peer_ranks"] = peers
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, devices, revision, job,
                                                 "--local-worker", runtime_trace=True, model=job["model"]["path"])

    def qualify(self, queries, recording, timeout=None):
        model = self.model(timeout)
        with TemporaryDirectory(dir=recording.directory.parent) as scratch:
            directory = Path(scratch)
            report = CostSamples.transport(recording, queries, model, directory)

            # Staged and physical graphs may differ internally. Use the normal evaluator on the measured wire operand.
            missing = CostSamples.uncovered(queries, report, model, directory)
        name = "samples/transport.json"
        answers = [{"status": "untriggered", "detail": missing[i]} if i in missing else {"status": "qualified", "asset": name}
                   for i in range(len(queries))]
        return ({name: report} if len(missing) < len(queries) else {}), answers


class LocalHostWork(LocalSchedulerWork):
    name = "sglang.ascend.scheduler_host_service"

    def __init__(self, arguments):
        super().__init__(arguments)
        self.selected_catalog = None

    @staticmethod
    def matches(query):
        return query.get("work") == "host" and query["domain"]["framework"] == "sglang" and "torch_npu" in query["domain"]

    def sources(self, deployment, rank):
        for directory in self.arguments.structure_session:
            receipt = json.loads((directory / "result.json").read_bytes())
            if (receipt.get("rule") not in {LocalSchedulerBatch.name, LocalWorkerBinding.name}
                    or receipt["status"] not in {"qualified", "incomplete"}):
                continue
            original = PublishedCapture(directory / "native")
            if (original.attempt_id != receipt["native_recording"] or original.inputs["rule"] != receipt["rule"]
                    or original.inputs["queries"] != receipt["queries"]):
                raise ValueError("structural training session does not own the qualified native recording")
            if original.inputs["deployment"] == deployment:
                yield from (worker for worker in WorkerCapture.observed(original) if worker.rank == rank)

    def prepare(self, queries, deployment, arguments, limit):
        requirements = CostRequirements()
        needed = {requirements.gap("query", q) for q in queries}
        image = NativeAttempt.image(arguments.source_image)["Id"]
        model = self.model(arguments.preparation_seconds)
        rank = WorkerDomain.rank(queries[0]["domain"])
        for worker in self.sources(deployment, rank):
            source = worker.recording
            if source.inputs["image"] != image:
                continue
            job = source.inputs["job"]
            if WorkerDomain.execution_domain(worker.read("facts")["environment"]) != queries[0]["domain"]:
                continue
            with TemporaryDirectory(dir=arguments.output.parent) as scratch:
                _, catalog = worker.catalog(model, Path(scratch))
            points = {point for _, point in requirements.observe_graph("source", catalog)}
            if needed <= points:
                self.selected_catalog = source.attempt_id, rank, catalog
                return SGLangAscendLaunch.local_attempt(
                    arguments, limit, deployment, source.inputs["physical_devices"], source.inputs["framework_revision"],
                    job, "--local-host", runtime_trace=True, host_timing=True, model=job["model"]["path"],
                    template_recording=source.attempt_id,
                )

        # A portable library has no raw profiles. Recreate only the live cut and time it in a separate worker.
        job, devices, revision = LocalSchedulerBatch.work_job(queries, deployment)
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, devices, revision, job, "--local-host-pair",
                                                 runtime_trace=True, host_timing=True, model=job["model"]["path"])

    def qualify(self, queries, recording, timeout=None):
        inputs = recording.inputs
        rank = WorkerDomain.rank(queries[0]["domain"])
        if "probes" in recording.manifest:
            original, timed = (WorkerCapture(recording, rank, name) for name in ("structure", "timing"))
        else:
            original = next((w for w in self.sources(inputs["deployment"], rank)
                             if w.recording.attempt_id == inputs["template_recording"]), None)
            if original is None:
                raise EvidenceGap("missing_local_structure_source", required_recording=inputs["template_recording"])
            timed = WorkerCapture(recording, rank)
        if self.selected_catalog and self.selected_catalog[:2] == (original.recording.attempt_id, rank):
            catalog = self.selected_catalog[2]
        else:
            with TemporaryDirectory(dir=recording.directory.parent) as scratch:
                _, catalog = original.catalog(self.model(timeout), Path(scratch))
        report = HostCostSamples.local(original, timed, catalog)
        return self.answers(queries, report, "samples/host.json")
