"""Record scheduler call boundaries on the existing lightweight serving path."""

from functools import partial
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns, process_time_ns, thread_time_ns

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors

from . import CaptureProbe


class BatchTiming:
    def __init__(self, window, aliases):
        self.window = window
        self.aliases = aliases
        self.calls = []

    @property
    def active(self):
        return self.window.phase == "source" and not self.window.drained

    def work(self, batch):
        if batch is None or not batch.reqs:
            return None

        members = BatchTensors.members(batch)
        for member in members:
            member["id"] = self.aliases[member["id"]]
            del member["slot"]
        return {"phase": BatchTensors.phase(batch), "members": members}

    @staticmethod
    def measure(original, *args, **kwargs):
        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        process_begin = process_time_ns()
        thread_begin = thread_time_ns()
        result = original(*args, **kwargs)
        thread_cpu = thread_time_ns() - thread_begin
        process_cpu = process_time_ns() - process_begin
        return result, {
            "begin_raw_ns": begin, "end_raw_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
            "thread_cpu_ns": thread_cpu, "process_cpu_ns": process_cpu,
        }

    def select(self, original, scheduler):
        if not self.active:
            return original(scheduler)

        batch, timing = self.measure(original, scheduler)
        self.calls.append({"call": "select", **timing, "work": self.work(batch)})
        return batch

    def invoke(self, call, original, scheduler, batch, *args, **kwargs):
        if not self.active:
            return original(scheduler, batch, *args, **kwargs)

        # Result processing can release native slots. Identify the batch before starting the clock.
        work = self.work(batch)
        result, timing = self.measure(original, scheduler, batch, *args, **kwargs)
        self.calls.append({"call": call, **timing, "work": work})
        return result

    def install(self):
        HookTable.group({
            symbols().scheduler_get_next_batch_to_run: self.select,
            symbols().scheduler_run_batch: partial(self.invoke, "run_batch"),
            symbols().scheduler_process_batch_result: partial(self.invoke, "process_result"),
        }).install()

    def document(self):
        return {"clock": "monotonic_raw_ns", "calls": self.calls}


def batch_timing(invocation):
    from ..native_session import NativeServingSession

    NativeServingSession(invocation.job, invocation.output, "latency-only", False, None, batch_timing=True).run()


probes = (CaptureProbe("batch-timing", "record lightweight scheduler call boundaries", batch_timing),)
