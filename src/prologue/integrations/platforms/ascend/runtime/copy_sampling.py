"""Time synchronous PyTorch tensor copies on one warmed, otherwise idle NPU."""

import json
from collections import defaultdict
from threading import get_native_id
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.integrations.platforms.ascend.analysis.services import SynchronousCopy
from prologue.integrations.platforms.ascend.runtime.host_profile import AscendRuntimeTiming
from prologue.runtime.io import write_json


class TensorCopySamples:
    directions = {"torch.Tensor.tolist": "device_to_host", "torch.Tensor.cpu": "device_to_host",
                  "torch.Tensor.to": "host_to_device"}
    implementation = "aclrtMemcpy"

    @classmethod
    def service(cls, caller):
        return {"operation": "local_copy", "caller": caller, "implementation": cls.implementation,
                "direction": cls.directions[caller], "resource": ["thread", "control"]}

    @staticmethod
    def now():
        return clock_gettime_ns(CLOCK_MONOTONIC_RAW)

    def sample(self, job, directory, revision):
        import torch

        from prologue.integrations.platforms.ascend.runtime.markers import AscendMarkers

        directory.mkdir(parents=True)
        start = self.now()
        actual = AscendMarkers.environment()
        if torch.npu.device_count() != 1 or any(actual[key] != value for key, value in job["runtime"].items()):
            raise ValueError("isolated copy requires the recorded runtime and exactly one visible NPU")

        # All operands are contiguous views of a preallocated buffer. Allocation and first use precede timing.
        operands = []
        for operand in job["operands"]:
            caller = operand["caller"]
            direction = self.directions[caller]
            copy = getattr(torch.Tensor, caller.rsplit(".", 1)[1])
            name, width = operand["dtype"], operand["element_bytes"]
            sizes = [point["bytes"] for point in operand["points"]]
            if width <= 0 or not sizes or any(type(size) is not int or size <= 0 or size % width for size in sizes):
                raise ValueError("copy sizes must be positive whole numbers of tensor elements")
            dtype = getattr(torch, name.removeprefix("torch."))
            upload = direction == "host_to_device"
            buffer = torch.ones(max(sizes) // width, dtype=dtype, device="cpu" if upload else "npu")
            buffer[1::2] = 0
            if buffer.element_size() != width or str(buffer.dtype) != name:
                raise ValueError("isolated operand differs from its observed dtype or element width")
            for size in sizes:
                row = {"caller": caller, "dtype": name, "bytes": size, "observations_ns": [], "plain_call_ns": [],
                       "timed_call_ns": [], "unmatched_calls": 0}
                tensor = buffer[:size // width]
                args = (tensor, "npu") if upload else (tensor,)
                operands.append((copy, args, row))
        AscendMarkers.synchronize()
        prepared = self.now()
        for copy, args, _ in operands:
            copy(*args)
        AscendMarkers.synchronize()
        warmed = self.now()

        windows = []
        readbacks = []
        path = directory / "runtime-timing.jsonl"
        timing = AscendRuntimeTiming()
        for enabled in (False, True):
            if enabled:
                timing.start()
            try:
                for repeat in range(7):
                    for copy, args, row in operands:
                        begin = self.now()
                        result = copy(*args)
                        end = self.now()
                        row["timed_call_ns" if enabled else "plain_call_ns"].append(end - begin)
                        if enabled:
                            windows.append((row, format(args[0].data_ptr(), "x"), begin, end))
                            if repeat == 6:
                                readbacks.append((row, args[0], result))
                        del result
            finally:
                if enabled:
                    timing.finish(path)
        observed = self.now()

        # Read the actual destination after timing stops, including the return value of Tensor.tolist.
        for row, source, result in readbacks:
            if isinstance(result, torch.Tensor):
                destination = "npu" if source.device.type == "cpu" else "cpu"
                row["verified_operand"] = (result.device.type == destination and result.dtype == source.dtype
                                           and torch.equal(result.cpu(), source.cpu()))
            else:
                row["verified_operand"] = row["caller"] == "torch.Tensor.tolist" and result == source.tolist()

        transfers = defaultdict(list)
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if (record["api"] == self.implementation and record["result"] == 0
                        and record["thread"] == get_native_id()):
                    transfers[record["source"], record["bytes"]].append(record)
        for row, address, begin, end in windows:
            matches = [call for call in transfers[address, row["bytes"]]
                       if begin <= call["begin_raw_ns"] <= call["end_raw_ns"] <= end
                       and SynchronousCopy.matches(call, address, row["bytes"], self.directions[row["caller"]])]
            if len(matches) == 1:
                row["observations_ns"].append(matches[0]["end_raw_ns"] - matches[0]["begin_raw_ns"])
            else:
                row["unmatched_calls"] += 1

        rows = [row for _, _, row in operands]
        write_json(directory / "services.json", rows, replace=False)
        write_json(directory / "capture.json", {
            "framework_revision": revision, "runtime": actual,
            "services": {row["caller"]: self.service(row["caller"]) for row in rows},
            "completion": "successful_synchronous_acl_return", "engine_starts": 0, "model_runner_starts": 0,
            "preparation_ns": prepared - start, "warmup_ns": warmed - prepared, "observation_ns": observed - warmed,
            "readback_ns": self.now() - observed,
            "measurements": "services.json",
        }, replace=False)
