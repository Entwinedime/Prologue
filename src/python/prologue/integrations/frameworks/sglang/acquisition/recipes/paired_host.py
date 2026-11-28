"""Acquire a CPU interval and an independently observed worker cut in separate processes."""

from dataclasses import replace

from prologue.integrations.bindings import RuleName

from ..contracts import WorkerDomain
from ..evidence import CostSamples
from ..host_samples import HostCostSamples
from ..readback import WorkerCapture
from .scheduler import LocalSchedulerBatch
from .services import LocalSchedulerWork


class PairedHostService(LocalSchedulerWork):
    name = RuleName("paired_scheduler_host_service")

    def matches(self, query):
        return query.get("work") == "host" and WorkerDomain.native_domain(query["domain"])

    def cut(self, queries, session, deployment, arguments):
        cut = LocalSchedulerBatch.work_job(queries, deployment)
        batch = cut.job["batch"]
        if "rank" in batch and not cut.job["single_device"]:
            batch["peer_ranks"] = sorted(map(int, queries[0]["domain"]["worker_roles"]))
        return replace(cut, worker="--local-host-pair", host_timing=True)

    def qualify(self, queries, recording, timeout=None):
        domain = queries[0]["domain"]
        requested = WorkerDomain.rank(domain)
        peers = recording.inputs["job"]["batch"].get("peer_ranks", ())
        invocations = LocalSchedulerBatch.invocations(queries)
        model = self.compiler(timeout)
        support = {}

        for rank in (requested, *(peer for peer in peers if peer != requested)):
            role = domain["worker_roles"][str(rank)]
            placed = WorkerDomain.execution_domain(WorkerDomain.bind_worker(domain, role))
            calls = [invocation | {"domain": placed} for invocation in invocations]
            original, timed = (WorkerCapture(recording, rank, name) for name in ("structure", "timing"))
            facts, _, structure = LocalSchedulerBatch.readback(calls, original)
            evidence, catalog = original.measure(model, recording.directory.parent)

            # Each rank contributes its own paired timings even when its schedule is shared.
            report = HostCostSamples.local(original, timed, catalog)
            if rank != requested:
                support[f"samples/host-rank-{rank}.json"] = report
                continue

            products, answers = self.answers(queries, report, "samples/host.json")
            if not products:
                return products, answers
            support.update({structure: facts, **CostSamples.device_support(evidence, catalog)})

        LocalSchedulerBatch.attach_support(products, answers, support)
        return products, answers


rules = (PairedHostService,)
