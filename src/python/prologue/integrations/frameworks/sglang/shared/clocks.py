"""Intercept decision clocks in the native ABI without changing Python's process clock."""

import importlib
import time
from contextvars import ContextVar
from functools import partial

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import symbols


class ClockNamespace:
    def __init__(self, owner, native, module) -> None:
        self.owner, self.native, self.module = owner, native, module

    def __getattr__(self, name):
        consumer = self.owner.consumer.get()
        if (
            consumer is not None
            and consumer.startswith(self.module + ".")
            and name in {"time", "monotonic", "perf_counter"}
        ):
            return partial(self.owner.read, name, consumer)
        return getattr(self.native, name)


class DecisionClocks:
    wait_queue_entry = "SchedulerReqTimeStats.set_wait_queue_entry_time"

    def __init__(self, calls) -> None:
        self.calls = calls
        self.consumer = ContextVar("native_decision_clock", default=None)

    def read(self, name, consumer):
        return self.calls.clock_read(getattr(time, name), {"kind": "clock", "clock": name, "consumer": consumer})

    def invoke(self, consumer, original, *args, **kwargs):
        previous = self.consumer.set(consumer)
        try:
            return original(*args, **kwargs)
        finally:
            self.consumer.reset(previous)

    def install(self):
        # These functions create or consume policy timestamps; metrics and background/watchdog clocks stay native.
        consumers = {
            symbols().scheduler_module: (
                "Scheduler._abort_on_running_timeout", "Scheduler._abort_on_waiting_timeout",
            ),
            symbols().req_time_stats: (
                self.wait_queue_entry, "SchedulerReqTimeStats.set_forward_entry_time",
            ),
            symbols().decode: ("DecodePreallocQueue._ensure_prefill_info",),
            symbols().conn: (
                "CommonKVSender._check_bootstrap_timeout", "CommonKVReceiver._check_waiting_timeout",
            ),
            symbols().mooncake_conn: (
                "MooncakeKVSender.__init__", "MooncakeKVReceiver.send_metadata",
            ),
        }
        for name, methods in consumers.items():
            module = importlib.import_module(name)
            module.time = ClockNamespace(self, module.time, module.__name__)
            for method in methods:
                target = module.__name__ + "." + method
                HookTable.around(target, partial(self.invoke, target))
