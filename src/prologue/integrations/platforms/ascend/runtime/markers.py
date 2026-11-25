"""Record real stream boundaries without synchronizing the serving path."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from inspect import signature
from typing import Any

import torch
import torch_npu

from prologue.integrations.platforms.ascend.runtime.host_profile import AscendSpecializations
from prologue.integrations.platforms.ascend.runtime.thread_activity import AscendThreadActivity


class AscendMarkers:
    @staticmethod
    def sample_clock(user_device: int) -> dict:
        acl = ctypes.CDLL("libascendcl.so")
        device_id = acl.aclrtGetLogicDevIdByUserDevId
        device_id.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        device_id.restype = ctypes.c_int
        device = ctypes.c_int32()
        status = device_id(user_device, ctypes.byref(device))
        if status:
            raise RuntimeError(f"Ascend clock device mapping failed: {status}")

        sample = ctypes.CDLL(None).prologue_sample_device
        sample.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        sample.restype = ctypes.c_int
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
        self.enable_native = bridge.prologue_set_runtime_enabled
        self.enable_native.argtypes = [ctypes.c_int]
        self.enable_native.restype = None
        self.enable_native(0)

        # These bounded CPU-only helpers must not release the GIL just to read a clock or exchange a thread-local value.
        bridge = ctypes.PyDLL(None)
        self.host_sample = bridge.prologue_sample_host
        self.host_sample.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.host_sample.restype = None
        self.native_service = bridge.prologue_set_service_scope
        self.native_service.argtypes = [ctypes.c_int64]
        self.native_service.restype = ctypes.c_int64

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
