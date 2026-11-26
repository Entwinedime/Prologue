"""Observe tensor provenance, completion signals and Gloo work at real PyTorch boundaries."""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from ctypes import CDLL, POINTER, Structure, c_int, c_size_t, c_uint64
from dataclasses import dataclass
from functools import partialmethod
from itertools import count
from typing import Any

import torch
import torch.distributed as distributed
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils.weak import WeakIdKeyDictionary

from prologue.common.observation import ScopeParameters, SemanticCapture
from prologue.integrations.runtimes.pytorch.shared import CpuTransfer


def tensor_leaves(tree):
    return (item for item in _pytree.tree_leaves(tree) if isinstance(item, torch.Tensor))


@dataclass
class StorageBoundary:
    lifetime: StorageWeakRef
    identity: int
    version: int = 0
    writer: int | None = None
    allocation: int | None = None
    exposed: bool = True
    buffer: str = ""


class TensorBoundaries(TorchDispatchMode):
    """Capture tensor storage and data-transfer edges where a PyTorch operation crosses a semantic scope."""

    def __init__(self, capture: SemanticCapture) -> None:
        super().__init__()
        self.capture = capture
        self.parameters = ScopeParameters(capture.scopes)
        self.tensors: Any = WeakIdKeyDictionary()
        self.shapes: Any = WeakIdKeyDictionary()
        self.storages: dict[tuple[int, str | None], StorageBoundary] = {}
        self.storage_ids = count()
        self.value_ids = count()
        self.namespace = ""

    @contextmanager
    def metadata(self, name="prologue.observer"):
        with self.capture.runtime.scope(name), torch._C.DisableTorchFunction(), _disable_current_modes():
            yield

    auxiliary = partialmethod(metadata, "prologue.auxiliary")

    def storage(self, tensor: torch.Tensor, region: str | None = None) -> StorageBoundary:
        native = tensor.untyped_storage()
        identity = native._cdata, region
        previous = self.storages.get(identity)
        if previous is None or previous.lifetime.expired():
            previous = StorageBoundary(StorageWeakRef(native), next(self.storage_ids))
            self.storages[identity] = previous
        return previous

    def written(self, tensor: torch.Tensor, region: str | None = None) -> None:
        storage = self.storage(tensor, region)
        owner = self.capture.current.get()

        # A version identifies an externally visible mutation epoch, not the number of hidden operators.
        if storage.exposed or storage.writer != owner:
            storage.version += 1
        storage.writer = owner
        storage.exposed = False

    def bind_buffer(self, tensor: torch.Tensor, name: str) -> None:
        storage = self.storage(tensor)
        if storage.buffer and storage.buffer != name:
            raise ValueError("native buffer aliases require one declared storage owner")
        storage.buffer = name

    def declare_shape(self, tensor: torch.Tensor, rule: list[Any], *, scope: int | None = None) -> None:
        if len(rule) != tensor.ndim:
            raise ValueError("declared boundary geometry has a different rank from the native tensor")
        scope = self.capture.current.get() if scope is None else scope
        self.shapes[tensor] = tuple(tensor.shape), self.parameters.bind(rule, scope)

    def declared_shape(self, tensor: torch.Tensor) -> list[Any] | None:
        declared = self.shapes.get(tensor)
        if declared is None:
            return None
        observed, rule = declared
        if observed != tuple(tensor.shape):
            raise ValueError("tensor geometry changed after its boundary declaration")
        return self.parameters.local(rule, self.capture.current.get())

    def shape_rule(self, tensor: torch.Tensor) -> list[Any]:
        rule = self.declared_shape(tensor)
        if rule is None:
            raise ValueError("boundary needs a shape relation from its producer")
        return rule

    @staticmethod
    def copy_source(tensor, caller, direction="device_to_host"):
        from_host = direction == "host_to_device"
        if (tensor.device.type == "cpu") == from_host and tensor.is_contiguous():
            return {"address": format(tensor.data_ptr(), "x"), "element_bytes": tensor.element_size(),
                    "caller": caller, "direction": direction}
        return None

    def local_copy(self, tensor, shape, caller, direction="device_to_host"):
        with self.metadata():

            # A cost operand inside the native call introduces no external storage or readiness edge.
            return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "shape_rule": shape,
                    "copy_source": self.copy_source(tensor, caller, direction)}

    def observe(self, tensor: torch.Tensor, shape: list[Any] | None, *, region: str | None = None,
                materializer: str | None = None) -> dict[str, Any]:
        with self.metadata():
            storage = self.storage(tensor, region)
            key = storage.identity, storage.version
            previous = self.tensors.get(tensor)

            if previous is None or previous[0] != key:
                previous = key, next(self.value_ids)
                self.tensors[tensor] = previous
            storage.exposed = True
            result = {
                "value": f"{self.namespace}value/{previous[1]}",
                "storage": f"{self.namespace}storage/{storage.identity}",
                "version": storage.version,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "device": tensor.device.type,
                "writer": storage.writer,
                "allocation": storage.allocation,
                "buffer": storage.buffer,
            }

            if shape is not None:
                self.declare_shape(tensor, shape)
                result["shape_rule"] = shape
                if materializer:
                    result["copy_source"] = self.copy_source(tensor, materializer)

            return result

    def retire(self) -> None:

        # Captured ports contain plain metadata; only live native storage needs mutation tracking between calls.
        self.storages = {identity: state for identity, state in self.storages.items() if not state.lifetime.expired()}

    def __torch_dispatch__(self, operation: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        keywords = {} if kwargs is None else kwargs

        # Keep weak input lifetimes for alias comparison; read-only weights need no persistent mutation record.
        lifetimes = (StorageWeakRef(tensor.untyped_storage()) for tensor in tensor_leaves((args, keywords)))
        incoming = {lifetime.cdata: lifetime for lifetime in lifetimes}
        mutated = []
        for index, parameter in enumerate(operation._schema.arguments):
            if parameter.alias_info is not None and parameter.alias_info.is_write:
                value = args[index] if index < len(args) else keywords.get(parameter.name)
                mutated.extend(tensor_leaves(value))

        result = operation(*args, **keywords)
        with torch._C.DisableTorchFunction(), _disable_current_modes():
            for tensor in mutated:
                self.written(tensor)

            # torch.tensor materializes storage before dispatching lift_fresh; its aliased argument is still newly
            # created.
            fresh = operation is torch.ops.aten.lift_fresh.default
            for tensor in tensor_leaves(result):
                state = self.storage(tensor)
                prior = incoming.get(state.lifetime.cdata)

                # An input may retire its storage during the native call, so addresses alone cannot establish aliasing.
                if state.writer is None and (fresh or prior is None or prior.expired()):
                    state.allocation = state.writer = self.capture.current.get()
                    state.exposed = False

        return result


class TensorUpload(TorchFunctionMode):
    """Read the actual CPU operand of one declared upload without moving its data."""

    def __init__(self, boundaries, parameter):
        self.boundaries = boundaries
        self.parameter = parameter
        self.operand = None

    def __torch_function__(self, operation, types, args=(), kwargs=None):
        result = operation(*args, **({} if kwargs is None else kwargs))
        if operation is torch.Tensor.to and args[0].device.type == "cpu" and result.device.type != "cpu":
            if self.operand is not None:
                raise ValueError("a declared tensor upload has multiple native operands")
            self.operand = self.boundaries.local_copy(args[0], [self.parameter], "torch.Tensor.to", "host_to_device")
        return result


class GatherLayout:
    @staticmethod
    def guard(shape: tuple[int, ...], rule: list[Any], dim: int, peers: int) -> list[Any]:
        dim %= len(shape)
        gathered = torch.empty((peers, *shape), device="meta")
        output_shape = (*shape[:dim], peers * shape[dim], *shape[dim + 1:])
        result = gathered.movedim(0, dim).reshape(output_shape)
        copied = result.untyped_storage()._cdata != gathered.untyped_storage()._cdata

        # Merging the moved group axis with the gathered axis needs a copy only when both strides matter.
        predicate = [
            "and", ["lt", 1, peers], ["lt", 1, ["mul", 1, *rule[:dim]]],
            ["lt", 1, rule[dim]], ["lt", 0, ["mul", 1, *rule]],
        ]
        return predicate if copied else ["eq", predicate, 0]


@dataclass
class ObservedSignal:
    point: int | None
    context: dict
    exported: bool = False


class NativeSignals:
    """Associate observed process-group completions with the original protocol call and its lifetime."""

    def __init__(self, current, scope, work_context, runtime=None) -> None:
        self.current = current
        self.scope = scope
        self.work_context = work_context
        self.runtime = runtime
        self.observed = ContextVar("signal_owner", default=None)
        self.started = False

    @property
    def active(self):
        return self.started and self.work_context.calls.enabled

    @contextmanager
    def observe(self):
        self.started = True
        token = self.observed.set(self.current.get())
        try:
            yield
        finally:
            self.observed.reset(token)

    def record(self, original, event, stream=None):
        if not self.active or (self.runtime is not None and self.runtime.recording.get()):
            return original(event, stream)
        if self.current.get() == 0:
            with self.scope(
                "signal_record", original, work=self.work_context.signal_work(stream), device=True, stream=stream,
            ) as call:
                result = self.record_event(original, event, stream)
                self.publish(call, event, "completion")

            return result

        if self.current.get() != self.observed.get():
            return original(event, stream)
        return self.record_event(original, event, stream)

    def record_event(self, original, event, stream):

        # Mark the native record itself; recording a second observer event would change the fence being exported.
        point_scope = nullcontext() if self.runtime is None else self.runtime.native_point(self.current.get(), "signal")
        with point_scope as point:
            result = original(event, stream)

        # The event owns its generation; native Event destruction does not reliably retire weak-key entries.
        event._prologue_signal = ObservedSignal(point, self.work_context.signal_work(stream)["context"])
        return result

    def publish(self, call, event, name):
        signal = event._prologue_signal
        signal.exported = True
        if signal.point is not None:
            call.signal(name, "output", signal.point)

    def query(self, original, event):
        signal = getattr(event, "_prologue_signal", None)
        if not self.active or signal is None or not signal.exported:
            return original(event)

        work = self.work_context.work()
        work["context"]["producer"] = signal.context

        # A query bounds completion; its return time is not the device's completion timestamp.
        with self.work_context.calls.read("event", original, work) as observed:
            if signal.point is not None:
                observed["point"] = signal.point
            result = original(event)
            observed["result"] = result

        return result

    def consume(self, device, original, event, stream=None):
        arguments = (event, stream) if device else (event,)
        signal = getattr(event, "_prologue_signal", None)
        if not self.active or signal is None or not signal.exported:
            return original(*arguments)
        if self.current.get() != 0:
            raise ValueError("an exported event wait must have its own action boundary")

        with self.scope(
            "signal_wait", original, work=self.work_context.wait_work(signal.context, device=device, stream=stream),
            bindings={"device_wait": device}, device=device, stream=stream,
        ) as call:
            if signal.point is not None:
                call.signal("completion", "input", signal.point)
            return original(*arguments)

    wait = partialmethod(consume, True)
    synchronize = partialmethod(consume, False)


class GlooSample(Structure):
    _fields_ = [
        ("post_ns", c_uint64), ("completion_begin_ns", c_uint64), ("completion_end_ns", c_uint64),
        ("slot", c_uint64), ("peer", c_int), ("completion_thread", c_int),
    ]

    def document(self):
        if not self.post_ns:
            return None
        completion = [self.completion_begin_ns, self.completion_end_ns] if self.completion_begin_ns else None
        return {"post_raw_ns": self.post_ns, "completion_raw_ns": completion, "slot": self.slot, "peer": self.peer,
                "completion_thread": self.completion_thread if completion is not None else None}


class GlooTiming:
    """Read the native Gloo probe at recorded CPU protocol boundaries."""

    clock = "monotonic_raw_ns"

    def __init__(self):
        runtime = CDLL(None)
        signatures = (
            ("begin_window", (), None), ("begin_post", (c_int,), c_uint64), ("end_post", (), None),
            ("capture", (POINTER(c_uint64), c_size_t, POINTER(GlooSample)), None),
        )
        for name, parameters, result in signatures:
            function = getattr(runtime, "prologue_gloo_" + name)
            function.argtypes, function.restype = parameters, result
            setattr(self, name, function)

    def document(self, posts):
        size = len(posts)
        tickets = (c_uint64 * size)(*(post.pop("transport_ticket", 0) for post in posts))
        samples = (GlooSample * size)()
        self.capture(tickets, size, samples)
        for post, sample in zip(posts, samples, strict=True):
            post["transport"] = sample.document()


@dataclass(eq=False)
class ProtocolWork:
    observer: CpuProtocol
    native: Any
    post: dict
    consumed: bool = False

    def wait(self, *args, **kwargs):
        if self.consumed:
            return self.native.wait(*args, **kwargs)
        interval = self.observer.interval()
        with self.observer.service_scope(interval):
            result = self.native.wait(*args, **kwargs)
        if result:
            self.consumed = True
            self.observer.publish(self.post, interval)
        return result


class CpuProtocol:
    def __init__(self, world, context, scope, service_scope, *, transport_timing=False) -> None:
        self.context = context
        self.scope = scope
        self.service_scope = service_scope
        self.world = world
        self.lock = threading.Lock()
        self.positions = {}
        self.posts = {}
        self.active = False
        self.timing = GlooTiming() if transport_timing else None

    def interval(self):
        execution = self.context.get()
        if execution is None:
            raise ValueError("Native CPU protocol observation requires its entering execution context")
        return {"thread": threading.get_native_id(), "scope": self.scope.get(), "execution": execution}

    def begin(self):
        with self.lock:
            self.start = list(self.positions.items())
            self.posts = {}
            if self.timing is not None:
                self.timing.begin_window()
            self.active = True

    def publish(self, post, consumed=None):
        with self.lock:
            if self.active:
                recorded = self.posts.setdefault(post["key"], {**post, "consumed": None})
                if consumed is not None:
                    recorded["consumed"] = consumed

    def invoke(self, name, transfer, group, original, *args, **kwargs):
        if distributed.get_backend(group) != "gloo":
            return original(*args, **kwargs)

        # Each PD instance has local PyTorch ranks. The recording uses the logical serving ranks.
        description = {**transfer, "source": self.world[transfer["source"]], "target": self.world[transfer["target"]]}
        route = CpuTransfer.route(description)
        with self.lock:
            position = self.positions.get(route, 0)
            self.positions[route] = position + 1

        post = {"key": (*route, position), "operation": name, "descriptor": description,
                "interval": self.interval()}

        # Keep preposted Work facts on its handle; a wait can cross the observation's beginning.
        if self.timing is not None and self.active:
            post["transport_ticket"] = self.timing.begin_post(description["receiving"])
        try:
            with self.service_scope(post["interval"]):
                result = original(*args, **kwargs)
        finally:
            if "transport_ticket" in post:
                self.timing.end_post()

        asynchronous = name in {"isend", "irecv"}
        self.publish(post, None if asynchronous else post["interval"])
        return ProtocolWork(self, result, post) if asynchronous else result

    def document(self):
        with self.lock:
            self.active = False
            posts = list(self.posts.values())
            result = {
                "backend": "gloo", "operations": list(CpuTransfer.operations),
                "start_positions": [{"route": route, "position": position} for route, position in self.start],
                "end_positions": [{"route": route, "position": position} for route, position in self.positions.items()],
                "posts": posts,
            }

            if self.timing is not None:
                self.timing.document(posts)
                result["transport_clock"] = self.timing.clock
            return result
