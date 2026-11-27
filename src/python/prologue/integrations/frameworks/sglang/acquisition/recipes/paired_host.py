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
        return replace(cut, worker="--local-host-pair", host_timing=True)

    def qualify(self, queries, recording, timeout=None):
        rank = WorkerDomain.rank(queries[0]["domain"])

        # Matching cuts identify the service. The structural pass must never supply its elapsed time.
        original, timed = (WorkerCapture(recording, rank, name) for name in ("structure", "timing"))
        invocations = LocalSchedulerBatch.invocations(queries)
        facts, _, structure = LocalSchedulerBatch.readback(invocations, original)
        evidence, catalog = original.measure(self.compiler(timeout), recording.directory.parent)
        report = HostCostSamples.local(original, timed, catalog)
        products, answers = self.answers(queries, report, "samples/host.json")

        # The already qualified structural cut can also answer later template queries.
        if products:
            support = {structure: facts, **CostSamples.device_support(evidence, catalog)}
            LocalSchedulerBatch.attach_support(products, answers, support)

        return products, answers


rules = (PairedHostService,)
