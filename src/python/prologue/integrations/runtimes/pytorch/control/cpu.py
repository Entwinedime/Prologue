"""Import native kernel declarations without loading their accelerator implementations."""

import sys
import types
from pathlib import Path


class DeclarationDriver:
    def get_benchmarker(self):
        return self.reject

    @staticmethod
    def reject(*args, **kwargs):
        raise RuntimeError("Native control must submit physical work to Prologue execution")


class CpuDeclarations:
    @staticmethod
    def install(kernel_python: Path) -> None:
        import triton

        triton.runtime.driver.set_active(DeclarationDriver())

        # The fixed upstream Python declarations are retained; only their binary loader is replaced.
        sys.path.insert(0, str(kernel_python))
        loader = types.ModuleType("sgl_kernel.load_utils")
        loader.__package__ = "sgl_kernel"
        loader._load_architecture_specific_ops = CpuDeclarations.load_declarations
        loader._preload_cuda_library = CpuDeclarations.load_declarations
        sys.modules[loader.__name__] = loader

    @staticmethod
    def load_declarations() -> None:
        pass


class MetadataWork:
    def __init__(self, peer, handle, tensor, wait_operation: str, *, context=None) -> None:
        self.peer = peer
        self.handle = handle
        self.tensor = tensor
        self.wait_operation = wait_operation
        self.finished = handle is None
        self.context = context

    def wait(self):
        if not self.finished:
            message = self.handle if self.context is None else {"key": self.handle, "execution": self.context.get()}
            if self.context is not None and message["execution"] is None:
                raise ValueError("CPU metadata wait requires its native protocol context")
            values = self.peer.exchange(self.wait_operation, message)
            if self.tensor is not None:
                import torch

                self.tensor.copy_(torch.tensor(values, dtype=self.tensor.dtype).reshape_as(self.tensor))
            self.finished = True
        return True
