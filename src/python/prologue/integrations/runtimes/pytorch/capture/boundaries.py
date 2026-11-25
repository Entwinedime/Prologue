"""Keep only boundary tensor identities and live mutation provenance."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils.weak import WeakIdKeyDictionary

from prologue.common.observation.capture import SemanticCapture
from prologue.common.observation.parameters import ScopeParameters


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
    def __init__(self, capture: SemanticCapture) -> None:
        super().__init__()
        self.capture = capture
        self.parameters = ScopeParameters(capture.scopes)
        self.tensors: Any = WeakIdKeyDictionary()
        self.shapes: Any = WeakIdKeyDictionary()
        self.storages: dict[tuple[int, str | None], StorageBoundary] = {}
        self.next_storage = 0
        self.next_value = 0
        self.namespace = ""

    @contextmanager
    def metadata(self) -> Iterator[None]:
        with self.capture.runtime.scope("prologue.observer"), torch._C.DisableTorchFunction(), _disable_current_modes():
            yield

    @contextmanager
    def auxiliary(self) -> Iterator[None]:
        with (
            self.capture.runtime.scope("prologue.auxiliary"),
            torch._C.DisableTorchFunction(),
            _disable_current_modes(),
        ):
            yield

    def storage(self, tensor: torch.Tensor, region: str | None = None) -> StorageBoundary:
        native = tensor.untyped_storage()
        identity = native._cdata, region
        previous = self.storages.get(identity)
        if previous is None or previous.lifetime.expired():
            previous = StorageBoundary(StorageWeakRef(native), self.next_storage)
            self.next_storage += 1
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
                previous = key, self.next_value
                self.next_value += 1
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
        incoming = {}
        for tensor in _pytree.tree_leaves((args, keywords)):
            if isinstance(tensor, torch.Tensor):
                lifetime = StorageWeakRef(tensor.untyped_storage())
                incoming[lifetime.cdata] = lifetime

        schema = operation._schema
        mutated = []
        if schema.is_mutable:
            for index, parameter in enumerate(schema.arguments):
                if parameter.alias_info is not None and parameter.alias_info.is_write:
                    value = args[index] if index < len(args) else keywords.get(parameter.name)
                    mutated.extend(item for item in _pytree.tree_leaves(value) if isinstance(item, torch.Tensor))

        result = operation(*args, **keywords)
        with torch._C.DisableTorchFunction(), _disable_current_modes():
            for tensor in mutated:
                self.written(tensor)

            # torch.tensor materializes storage before dispatching lift_fresh; its aliased argument is still newly
            # created.
            fresh = operation is torch.ops.aten.lift_fresh.default
            for tensor in _pytree.tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    state = self.storage(tensor)
                    prior = incoming.get(state.lifetime.cdata)

                    # Rebinding an input can retire its old storage during the call; an address alone is not a lifetime.
                    if state.writer is None and (fresh or prior is None or prior.expired()):
                        state.writer = self.capture.current.get()
                        state.allocation = state.writer
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
