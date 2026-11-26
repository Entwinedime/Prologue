"""Record native markers, thread activity and calibrated host service windows."""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from inspect import signature
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns
from typing import Any

import torch
import torch_npu

from prologue.common.io import write_json


def native_function(library, symbol, result, *operands):
    function = getattr(library, symbol)
    function.argtypes, function.restype = operands, result
    return function


class WorkerThread(ctypes.Structure):
    _fields_ = [("local", ctypes.c_int), ("kernel", ctypes.c_uint32)]


class TraceLayout(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("id", "first", "second")]


class WakeLayout(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("id", "recipient", "producer", "flags")]


class AscendThreadActivity:
    trace_events = Path("/opt/prologue/trace-events")
    host_tasks = Path("/opt/prologue/host-proc/self/task")
    kernel_fields = (
        ("sched/sched_switch", (("prev_pid", 4), ("prev_state", 8))),
        ("syscalls/sys_enter_futex", (("uaddr", 8), ("op", 8))),
        ("syscalls/sys_exit_futex", (("ret", 8),)),
    )

    def __init__(self):
        bridge = ctypes.PyDLL(None)
        self.open = native_function(
            bridge, "prologue_thread_activity_start", ctypes.c_void_p,
            ctypes.POINTER(WorkerThread), ctypes.c_size_t, ctypes.POINTER(TraceLayout), ctypes.POINTER(WakeLayout),
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        )
        self.close = native_function(
            bridge, "prologue_thread_activity_stop", ctypes.c_int,
            ctypes.c_void_p, ctypes.c_char_p,
        )

    @classmethod
    def layout(cls, event, layout_type, fields):
        source = cls.trace_events / event
        observed = {name: (int(offset), int(size)) for name, offset, size in re.findall(
            r"field:[^;]*?\b(\w+);\s+offset:(\d+);\s+size:(\d+);", (source / "format").read_text(),
        )}
        if observed.get("common_type") != (0, 2) or any(observed.get(name, (0, 0))[1] != size for name, size in fields):
            raise ValueError(f"unsupported worker tracepoint layout: {event}")
        return layout_type(int((source / "id").read_text()), *(observed[name][0] for name, _ in fields))

    @classmethod
    def owned_threads(cls):
        threads = []
        for task in cls.host_tasks.iterdir():
            try:
                status = (task / "status").read_text()
            except FileNotFoundError:
                continue
            namespace, = (line for line in status.splitlines() if line.startswith("NSpid:"))
            values = namespace.split()
            threads.append(WorkerThread(int(values[-1]), int(values[1])))
        return threads

    @staticmethod
    def online_cpus():
        cpus = []
        for interval in Path("/sys/devices/system/cpu/online").read_text().strip().split(","):
            first, _, last = interval.partition("-")
            cpus.extend(range(int(first), int(last or first) + 1))
        return (ctypes.c_int * len(cpus))(*cpus)

    def start(self, *, detailed=False):
        layouts, wake, cpus = None, None, ()
        if detailed:

            # The host proc view is used only for this worker's kernel thread identities.
            selected = self.owned_threads()
            layouts = (TraceLayout * len(self.kernel_fields))(
                *(self.layout(event, TraceLayout, fields) for event, fields in self.kernel_fields),
            )
            wake = self.layout("sched/sched_waking", WakeLayout, (("pid", 4), ("common_pid", 4), ("common_flags", 1)))
            cpus = self.online_cpus()
        else:
            selected = [WorkerThread(int(task.name), 0) for task in Path("/proc/self/task").iterdir()]

        threads = (WorkerThread * len(selected))(*selected)
        self.handle = self.open(threads, len(threads), layouts, ctypes.byref(wake) if wake is not None else None,
                                cpus if detailed else None, len(cpus))
        if not self.handle:
            raise RuntimeError("cannot open CPU activity observation for the native worker")

    def finish(self, output):
        if self.close(self.handle, os.fsencode(output)):
            raise RuntimeError("cannot publish native CPU activity evidence")

    @contextmanager
    def record(self, output):
        self.start(detailed=True)
        try:
            yield
        finally:
            self.finish(output)


class AscendSpecializations:
    def __init__(self) -> None:
        from triton.runtime.jit import JITFunction

        self.runtime = JITFunction
        self.started = False
        self.loads = []

    def start(self) -> None:
        self.previous_hook = self.runtime.cache_hook
        self.runtime.cache_hook = self.loaded
        self.started = True

    def loaded(self, **event):
        function = event["fn"]
        self.loads.append({
            "kernel": f"{function.module}.{function.name}",
            "specialization": str(event["key"]),
            "raw_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
        })
        return self.previous_hook(**event) if self.previous_hook is not None else False

    def finish(self) -> list[dict]:
        self.runtime.cache_hook = self.previous_hook
        return self.loads


class AscendRuntimeTiming:
    def __init__(self):
        bridge = ctypes.PyDLL(None)
        self.start = native_function(bridge, "prologue_runtime_timing_start", None)
        self.stop = native_function(bridge, "prologue_runtime_timing_stop", ctypes.c_int, ctypes.c_char_p)

    def finish(self, path):
        if self.stop(os.fsencode(path)):
            raise RuntimeError("cannot publish native runtime timing evidence")


class AscendHostProfile:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.started = False
        self.specializations = AscendSpecializations()
        self.activity = AscendThreadActivity()
        self.runtime = AscendRuntimeTiming()
        self.host_clock = native_function(
            ctypes.PyDLL(None), "prologue_sample_host_clock", None,
            ctypes.POINTER(ctypes.c_uint64),
        )

    def clock_sample(self):
        values = (ctypes.c_uint64 * 3)()
        self.host_clock(values)
        return list(values)

    def start(self) -> None:
        from torch.autograd import _enable_record_function
        from torch_npu import profiler

        self.profiler = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(self.directory / "host-profile"), analyse_flag=False),
        )
        self.begin_clock = self.clock_sample()
        self.profiler.start()

        # The SDK still emits queue marks with ATen range callbacks disabled.
        _enable_record_function(False)
        self.activity.start()
        self.specializations.start()
        self.runtime.start()
        self.started = True

    def finish(self) -> dict:
        from torch.autograd import _enable_record_function

        specializations = self.specializations.finish()
        runtime = self.directory / "runtime-timing.jsonl"
        activity = self.directory / "thread-activity.jsonl"
        with ExitStack() as cleanup:
            cleanup.callback(_enable_record_function, True)
            cleanup.callback(self.profiler.stop)
            cleanup.callback(self.activity.finish, activity)
            self.runtime.finish(runtime)

        profile, = (self.directory / "host-profile").glob("*_ascend_pt")
        queues = self.directory / "host-queues.json"
        write_json(queues, self.queue_intervals(profile, (self.begin_clock, self.clock_sample())), replace=False)
        return {
            "profile": str(profile.relative_to(self.directory)), "queues": queues.name,
            "activity": activity.name, "runtime": runtime.name,
            "specialization_loads": specializations,
        }

    @staticmethod
    def queue_intervals(profile: Path, clock_samples) -> dict:
        from torch_npu.profiler.analysis._profiler_config import ProfilerConfig
        from torch_npu.profiler.analysis.prof_parse._fwk_file_parser import FwkFileParser

        configuration = ProfilerConfig()
        configuration.load_info(str(profile))
        enqueues, dequeues = FwkFileParser(str(profile)).get_task_queue_data()

        # Queue records use the SDK startup counter pair. Bound its common offset with independent RAW readings.
        corrections = []
        for counter, begin, end in clock_samples:
            sdk_ns = configuration.get_timestamp_from_syscnt(counter)
            corrections.append((begin - sdk_ns, end - sdk_ns))

        lower = max(begin for begin, _ in corrections)
        upper = min(end for _, end in corrections)
        if lower > upper:
            raise ValueError("CPU queue timestamps have no common host clock calibration")

        offset_ns = configuration.get_local_time(0) - (lower + upper) // 2

        def interval(record):
            begin_ns = record.ts - offset_ns
            return [record.tid, begin_ns, begin_ns + record.dur]

        produced = {record.corr_id: record for record in enqueues}
        consumed = {record.corr_id: record for record in dequeues}
        if len(produced) != len(enqueues) or len(consumed) != len(dequeues):
            raise ValueError("host queue evidence repeats a native correlation")

        return {
            "clock": "monotonic_raw_ns",
            "clock_samples": clock_samples,
            "raw_correction_ns": [lower, upper],
            "unpaired": len(produced.keys() ^ consumed.keys()),
            "commands": [
                [command.origin_name, interval(produced[key]), interval(command)]
                for key, command in consumed.items() if key in produced
            ],
        }


class AscendMarkers:
    @staticmethod
    def sample_clock(user_device: int) -> dict:
        acl = ctypes.CDLL("libascendcl.so")
        device_id = native_function(
            acl, "aclrtGetLogicDevIdByUserDevId", ctypes.c_int,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        )
        device = ctypes.c_int32()
        status = device_id(user_device, ctypes.byref(device))

        if status:
            raise RuntimeError(f"Ascend clock device mapping failed: {status}")

        sample = native_function(
            ctypes.CDLL(None), "prologue_sample_device", ctypes.c_int,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
        )
        values = (ctypes.c_uint64 * 3)()

        # The native helper brackets only the actual queries; Python releases the GIL outside that interval.
        status = sample(device.value, values)
        if status:
            raise RuntimeError(f"Ascend native clock acquisition failed: {status}")
        counter, begin, end = values
        return {"device": device.value, "counter": counter, "begin_raw_ns": begin, "end_raw_ns": end}

    def __init__(self, *, detailed_timing: bool) -> None:
        self.points: list[dict[str, Any]] = []
        self.stream_events: dict[Any, Any] = {}
        self.lock = threading.Lock()
        self.recording = ContextVar("observer_point", default=False)
        self.clock_samples = []
        self.observing = False
        self.intervals = []
        self.activity = None
        self.specializations = AscendSpecializations() if detailed_timing else None

        bridge = ctypes.CDLL(None)
        self.enable_native = native_function(bridge, "prologue_set_runtime_enabled", None, ctypes.c_int)
        self.enable_native(0)

        # These bounded CPU-only helpers must not release the GIL just to read a clock or exchange a thread-local value.
        bridge = ctypes.PyDLL(None)
        self.host_sample = native_function(bridge, "prologue_sample_host", None, ctypes.POINTER(ctypes.c_uint64))
        self.native_service = native_function(bridge, "prologue_set_service_scope", ctypes.c_int64, ctypes.c_int64)

    @contextmanager
    def window(self, activity=None):
        observer = nullcontext() if activity is None else AscendThreadActivity().record(activity)
        if self.specializations is not None:
            self.specializations.start()

        try:
            with observer:
                self.clock_samples.append(self.sample_clock(torch_npu.npu.current_device()))
                self.observing = True
                try:
                    with self.scope("prologue.window"):
                        self.enable_native(1)
                        try:
                            yield
                        finally:
                            self.enable_native(0)
                finally:
                    self.observing = False
                    self.clock_samples.append(self.sample_clock(torch_npu.npu.current_device()))
        finally:
            if self.specializations is not None:
                self.specializations.finish()

        self.activity = activity

    @staticmethod
    def synchronize():
        torch_npu.npu.synchronize()

    @staticmethod
    def environment():
        return {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "device": torch_npu.npu.get_device_name(), "device_index": torch_npu.npu.current_device()}

    def timing_evidence(self):

        # A submitted operation can finish after the host window closes. Every caller exports after its device fence.
        drained = self.sample_clock(torch_npu.npu.current_device())
        result = {"device_clock": [*self.clock_samples, drained], "host_observation": self.host_observation()}
        if self.specializations is not None:
            result["specialization_loads"] = self.specializations.loads
        return result

    def checkpoint(self) -> None:
        if not self.observing:
            return
        elapsed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - self.clock_samples[-1]["end_raw_ns"]
        if elapsed_ns < 1_000_000_000:
            return

        # Sampling cadence controls observation overhead; the actual independent snapshots define the clock segments.
        with self.scope("prologue.observer"):
            self.clock_samples.append(self.sample_clock(torch_npu.npu.current_device()))

    def timestamp(self, record, side):
        sample = (ctypes.c_uint64 * 2)()
        self.host_sample(sample)
        record[f"{side}_counter"], record[f"{side}_raw_ns"] = sample

    def host_observation(self):
        result = {"process": os.getpid(), "intervals": self.intervals}
        if self.activity is not None:
            result["activity"] = self.activity.name
        return result

    @contextmanager
    def scope(self, name: str):
        if not self.observing:
            yield
            return
        record = {"name": name, "thread": threading.get_native_id()}
        self.intervals.append(record)
        with self.service_scope(record):
            yield

    @contextmanager
    def service_scope(self, interval):

        # Ordering uses the same CPU counter as SDK enqueue/API records; measurements stay in RAW time.
        self.timestamp(interval, "begin")
        try:
            yield
        finally:
            self.timestamp(interval, "end")

    @contextmanager
    def service_execution(self, scope: int, execution: dict):

        # Driver submissions carry the caller identity even on native executor threads without Torch callbacks.
        previous = self.native_service(scope)
        execution["thread"] = threading.get_native_id()
        try:
            with self.service_scope(execution):
                yield
        finally:
            self.native_service(previous)

    @contextmanager
    def native_point(self, scope: int, side: str):
        with self.lock:
            identity = len(self.points)
            self.points.append({"scope": scope, "side": side})
        token = self.recording.set(True)
        try:
            with self.scope(f"prologue.point/{identity}"):
                yield identity
        finally:
            self.recording.reset(token)

    def point(self, scope: int, side: str, stream=None) -> int:
        stream = torch_npu.npu.current_stream() if stream is None else stream

        # Nobody waits on observer handles; each repeated record has its own SDK measurement and point identity.
        with self.native_point(scope, side) as identity, self.lock:
            event = self.stream_events.get(stream)
            if event is None:
                event = torch_npu.npu.Event()
                self.stream_events[stream] = event
            event.record(stream)
        return identity

    def release(self) -> None:
        with self.scope("prologue.observer"), self.lock:
            self.stream_events.clear()


class AscendProfiling:
    @staticmethod
    def configuration(original, configuration, *args, **kwargs):
        arguments = signature(original).bind(configuration, *args, **kwargs)

        # Keep the native trace level and HCCL/API data; no structural consumer reads AI Core counters.
        arguments.arguments["aic_metrics"] = torch_npu.profiler.AiCMetrics.AiCoreNone
        return original(*arguments.args, **arguments.kwargs)

    @staticmethod
    def handler(original, directory, *args, **kwargs):

        # Native recording must close before CPU-only SDK export and graph production begin.
        kwargs["analyse_flag"] = False
        return original(directory, *args, **kwargs)

    @classmethod
    def hooks(cls):
        return {
            "torch_npu.profiler._ExperimentalConfig.__init__": cls.configuration,
            "torch_npu.profiler.tensorboard_trace_handler": cls.handler,
        }
