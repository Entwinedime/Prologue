"""Observe the native PD receiver copying each admitted request's page indices."""

import torch
from torch.overrides import TorchFunctionMode


class ReceiverTableCopies(TorchFunctionMode):
    def __init__(self, owner, batch, work):
        self.owner = owner
        self.table = batch.req_to_token_pool.req_to_token
        self.slots = [req.req_pool_idx for req in batch.reqs]
        self.work = work
        self.target = None
        self.writes = []

    def __torch_function__(self, operation, types, args=(), kwargs=None):
        keywords = {} if kwargs is None else kwargs
        if operation is not torch.Tensor.__setitem__:
            return operation(*args, **keywords)
        target, index, source = args
        if (
            not isinstance(source, torch.Tensor)
            or source.untyped_storage()._cdata != self.table.untyped_storage()._cdata
        ):
            return operation(*args, **keywords)

        with self.owner.tensors.metadata():
            if (not isinstance(index, slice) or target.ndim != 1 or target.stride(0) != 1
                    or source.ndim != 1 or source.stride(0) != 1):
                raise ValueError("receiver copy requires contiguous source and destination slices")
            slot, begin = divmod(source.storage_offset() - self.table.storage_offset(), self.table.stride(0))
            start, end, step = index.indices(target.numel())
            count = source.numel()
            if step != 1 or end - start != count or begin + count > self.table.size(1):
                raise ValueError("receiver copy exceeds its row or destination")
            view = target[index]

        position = len(self.writes)
        if position >= len(self.slots) or self.slots[position] != slot or (position and target is not self.target):
            raise ValueError("receiver copy does not follow the admitted rows into one destination")
        member = self.work["members"][position]
        work = {key: self.work[key] for key in ("phase", "rank")}
        work["members"] = [member]

        # Bracket only the native assignment. Reading the resulting KV and synchronizing it belong to qualification.
        with self.owner.capture.call("row_copy", "request_to_token.row_copy", "torch.Tensor.__setitem__",
                                     parameters={"elements": count}, arguments={"elements": "new_tokens"},
                                     work=work, device=True) as call:
            call.record["row_call"] = {"member": position, "parent_port": "cache_locations",
                                       "source_port": "source", "destination_port": "destination"}
            call.port("source", "input", self.owner.tensors.observe(source, ["elements"]), use="state")
            result = operation(*args, **keywords)
            call.port("destination", "output", self.owner.tensors.observe(view, ["elements"]), use="device")

        self.target = target
        self.writes.append({"slot": slot, "source": [begin, begin + count], "target": [start, end]})
        return result

    def readback(self, output):
        if len(self.writes) != len(self.slots) or (self.writes and self.target is not output):
            raise ValueError("receiver did not publish the destination filled by its admitted rows")
        return {"slots": self.slots, "writes": self.writes}
