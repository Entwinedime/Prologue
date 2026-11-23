"""Record a hierarchy of observed calls and the tensor objects crossing their ports."""

from contextlib import contextmanager
from threading import get_native_id
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns
from weakref import ref


class Calls:
    def __init__(self, annotate):
        self.annotate = annotate
        self.reset()

    def reset(self):
        self.rows = []
        self.stack = []
        self.tensors = {}
        self.next_tensor = 0

    @contextmanager
    def scope(self, operation, implementation, site, parameters):
        identity = len(self.rows)
        row = {
            "id": identity,
            "parent": self.stack[-1] if self.stack else None,
            "operation": operation,
            "implementation": implementation,
            "site": site,
            "parameters": parameters,
            "inputs": [],
            "outputs": [],
            "thread": get_native_id(),
        }
        self.rows.append(row)
        self.stack.append(identity)
        try:
            with self.annotate(f"prologue.scope.{identity}"):
                row["begin_ns"] = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
                yield row
                row["end_ns"] = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        finally:
            self.stack.pop()

    def ports(self, **tensors):
        result = []
        for name, tensor in tensors.items():
            if tensor is None:
                continue

            # Observe identity without retaining a tensor or tracing its storage address.
            key = id(tensor)
            previous = self.tensors.get(key)
            if previous is None or previous[0]() is not tensor:
                previous = ref(tensor), self.next_tensor
                self.tensors[key] = previous
                self.next_tensor += 1

            result.append({
                "name": name, "tensor": previous[1], "dtype": str(tensor.dtype),
                "shape": list(tensor.shape), "stride": list(tensor.stride()), "device": tensor.device.type,
            })
        return result
