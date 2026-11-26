"""Connect real metadata collectives and point-to-point calls to their scheduler boundaries."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.shared.calls import ActionCall
from prologue.integrations.runtimes.pytorch.shared import CpuTransfer


class HostReads:
    method = "torch.Tensor.tolist"

    def __init__(self, calls) -> None:
        self.calls = calls

    @staticmethod
    def result_inputs(result):
        tensor = result.next_token_ids
        return {} if tensor is None else {"next_token_ids": (tensor, ["members"])}

    def invoke(self, original, tensor, *args, **kwargs):
        calls = self.calls
        frame = calls.current.get()
        if not calls.enabled or frame is None:
            return original(tensor, *args, **kwargs)

        for name, (value, shape) in frame.host_inputs.items():
            if value is not tensor:
                continue
            work = {**frame.work, "context": {**frame.work["context"], "host_input": name}}
            boundary = ActionCall(
                "host_read",
                dict(frame.parameters),
                work,
                inputs={name: (tensor, shape)},
                condition={"input": name, "method": "tolist"},
                input_use="host",
                materializer=self.method,

                # This adapter-owned call has a published identity independent of its Python package layout.
                implementation="prologue.integrations.frameworks.sglang.host_reads.HostReads.invoke",
            )

            # Only this actual read owns readiness and waiting; unrelated tolist calls remain native.
            with calls.separate(boundary.operation, work), calls.boundary(self.invoke, boundary):
                return original(tensor, *args, **kwargs)

        return original(tensor, *args, **kwargs)

    def install(self):
        HookTable.around(self.method, self.invoke)


class MetadataCollectives:
    def __init__(self, calls) -> None:
        self.calls = calls

    @staticmethod
    def snapshot(arguments, root):
        import torch.distributed as distributed

        if "input_tensor" in arguments:
            inputs, outputs = arguments["input_tensor"], arguments["output_tensor"]
        else:
            inputs = outputs = arguments["tensor"]

        # A broadcast receiver's pre-call buffer is uninitialized storage, not an operand.
        values = (
            [] if root is not None and distributed.get_rank(arguments["group"]) != root else inputs.flatten().tolist()
        )
        return {"input": values}, outputs

    def invoke(self, name, signature, original, *args, **kwargs):
        calls = self.calls
        parent = calls.current.get()
        if not calls.enabled or parent is None:
            return original(*args, **kwargs)

        import torch.distributed as distributed

        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        tensor = values["input_tensor" if name == "all_gather_into_tensor" else "tensor"]
        group = distributed.group.WORLD if values["group"] is None else values["group"]

        if not calls.groups.is_cpu(group):
            return calls.device_collective(original, values)

        # Empty Gloo reductions and singleton groups do not impose a peer rendezvous.
        if distributed.get_world_size(group) == 1 or (name == "all_reduce" and tensor.numel() == 0):
            return original(*args, **kwargs)
        if values["async_op"]:
            raise ValueError("CPU metadata collectives require blocking completion")

        members = distributed.get_process_group_ranks(group)
        root = None
        if name == "broadcast":
            source = values.get("group_src")
            root = members.index(values["src"]) if source is None else source

        collective = {
            "group": calls.groups.name(group), "operation": name, "dtype": str(tensor.dtype), "root": root,
        }
        if name == "all_reduce":
            collective["reduction"] = str(values["op"])

        work = {**parent.work, "context": {**parent.work.get("context", {}), "collective": collective}}
        boundary = ActionCall(
            "metadata_collective", {"elements": tensor.numel()}, work, condition={"collective": collective},
        )

        # Submit preceding CPU work before rendezvous; the native call remains the sole owner of metadata values.
        with calls.separate(boundary.operation, work):
            return calls.collective(original, boundary, values)

    def install(self):
        import torch.distributed as distributed


        for name in ("broadcast", "all_gather_into_tensor", "all_reduce"):
            hook = partial(self.invoke, name, inspect.signature(getattr(distributed, name)))
            HookTable.around(f"torch.distributed.{name}", hook)


class MetadataWork:
    def __init__(self, owner, native, operation, transfer, group) -> None:
        self.owner, self.native = owner, native
        self.operation, self.transfer, self.group = operation, transfer, group

    def wait(self, *args, **kwargs):
        with self.owner.boundary(self.operation + "_wait", self.transfer, self.group):
            return self.native.wait(*args, **kwargs)


class MetadataTransfers:
    def __init__(self, calls, protocol=None) -> None:
        self.calls = calls
        self.protocol = protocol
        self.transfer = CpuTransfer()
        self.in_transfer = ContextVar("native_metadata_transfer", default=False)

    @contextmanager
    def boundary(self, operation, transfer, group):
        calls = self.calls
        active = calls.protocol_enabled
        parent = calls.current.get()
        work = {}

        if active:
            work = calls.scheduler_work() if parent is None else dict(parent.work)
            transfer = {**transfer, "group": calls.groups.name(group)}
            work["context"] = {**work["context"], "cpu_transfer": CpuTransfer.identity(transfer)}

        # Runtime services own protocol time; the caller must neither overlap it nor expand it as template work.
        with calls.separate("cpu_" + operation, work):
            previous = self.in_transfer.set(True)
            context = calls.communicating.set({"operation": operation, "in_flow": active, "work": work})
            try:
                yield
            finally:
                calls.communicating.reset(context)
                self.in_transfer.reset(previous)

    def invoke(self, name, signature, original, *args, **kwargs):

        # Blocking send/recv can call their async counterpart. Observe the outer native call exactly once.
        if self.in_transfer.get():
            return original(*args, **kwargs)
        native = self.transfer.native(name, signature, args, kwargs)
        if native is None:
            return original(*args, **kwargs)
        transfer, group = native
        run = original if self.protocol is None else partial(self.protocol.invoke, name, transfer, group, original)
        with self.boundary(name, transfer, group):
            result = run(*args, **kwargs)
        return MetadataWork(self, result, name, transfer, group) if name in {"isend", "irecv"} else result

    def install(self):
        for path, hook in CpuTransfer.hooks(self.invoke):
            HookTable.around(path, hook)
