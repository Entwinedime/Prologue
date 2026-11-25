"""Hand native CPU protocol intervals to runtime services, outside template CPU work."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.runtimes.pytorch.shared import CpuTransfer


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
