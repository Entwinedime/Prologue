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

    @staticmethod
    def project(rows):
        # Trace markers keep raw identities while the structural call tree keeps only materialized calls.
        omitted = {row["id"] for row in rows if row["operation"] == "capture_auxiliary"}
        completed = {}
        for row in rows:
            if row["operation"] == "collective_wait":
                post = rows[row["parameters"]["post"]]
                if (post["operation"] != "collective" or post["parent"] != row["parent"]
                        or post["id"] + 1 != row["id"] or not row["completed"]):
                    raise ValueError("a collective cut requires its immediate native completion wait")
                completed[row["id"]] = post["id"]

        retained = [row for row in rows if row["id"] not in omitted and row["id"] not in completed]
        identities = {row["id"]: index for index, row in enumerate(retained)}
        parameter_links = {"pipeline_prepare": "batch", "pipeline_process": "batch", "transfer_wait": "post",
                           "signal_record": "producer", "signal_wait": "producer"}
        projected = []
        for row in retained:
            parent = row["parent"]
            operation = row["operation"]
            call = {**row, "id": identities[row["id"]], "parent": identities[parent] if parent is not None else None}
            parameter = parameter_links.get(operation)
            if parameter is not None and row["parameters"][parameter] is not None:
                call["parameters"] = {**row["parameters"], parameter: identities[row["parameters"][parameter]]}
            if operation == "pipeline_launch":
                call["batch"] = identities[row["batch"]]
            if operation == "experts":

                # The model-layer number identifies the invocation. The route binding carries it at the root.
                call["parameters"] = {}
            projected.append(call)
        return projected, identities, omitted, completed
