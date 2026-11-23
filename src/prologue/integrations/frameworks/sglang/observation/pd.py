"""Observe Decode admission and real page-table copies at the prebuilt boundary."""

import json
import os
from contextlib import nullcontext
from itertools import count
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

import torch
from torch.autograd.profiler import record_function
from torch.overrides import TorchFunctionMode


class PageCopies(TorchFunctionMode):
    def __init__(self, batch, markers, detailed):
        self.pool = batch.req_to_token_pool.req_to_token
        self.markers = markers
        self.detailed = detailed
        self.rows = {}
        self.sources = {}
        self.targets = []
        self.records = []

    @staticmethod
    def extent(tensor, index):
        start, end, step = index.indices(tensor.shape[0])
        if step != 1:
            raise ValueError("page-table copy requires a contiguous slice")
        return [start, end]

    def __torch_function__(self, operation, types, args=(), kwargs=None):
        keywords = {} if kwargs is None else kwargs
        if operation is torch.Tensor.__setitem__ and id(args[2]) in self.sources:
            return self.copy(operation, args, keywords)

        value = operation(*args, **keywords)
        if operation is torch.Tensor.__getitem__:
            tensor, index = args
            if tensor is self.pool and isinstance(index, int):
                self.rows[id(value)] = (value, index)
            elif id(tensor) in self.rows and isinstance(index, slice):
                _, slot = self.rows[id(tensor)]
                self.sources[id(value)] = (value, slot, self.extent(tensor, index))
        return value

    def copy(self, operation, arguments, keywords):
        target, index, operand = arguments
        _, slot, source = self.sources[id(operand)]
        marker = next(self.markers) if self.detailed else None
        context = nullcontext() if marker is None else record_function(f"prologue.scope.{marker}")
        with context:
            value = operation(*arguments, **keywords)

        self.targets.append(target)
        self.records.append({
            "slot": slot, "source": source, "target": self.extent(target, index),
            "source_dtype": str(operand.dtype), "target_dtype": str(target.dtype),
            "source_device": operand.device.type, "target_device": target.device.type, "marker": marker,
        })
        return value


class PDObservation:
    def __init__(self, directory, specification):
        self.directory = directory
        self.configuration_id = specification["configuration_id"]
        self.detail = set(specification["detail_requests"])
        self.requests = self.detail | set(specification["parent_requests"])
        self.markers = count()
        self.prepared = None
        self.domain_written = False

    def record_domain(self, scheduler):
        args = scheduler.server_args
        domain = {
            "framework_revision": os.environ["PROLOGUE_SGLANG_REVISION"],
            "model": {"path": args.model_path, "dtype": args.dtype, "load_format": args.load_format},
            "configuration": {
                "id": self.configuration_id, "disaggregation_mode": args.disaggregation_mode,
                "tp": args.tp_size, "attention_backend": args.attention_backend, "page_size": scheduler.page_size,
                "max_running_requests": scheduler.max_running_requests,
                "max_total_tokens": scheduler.max_total_num_tokens,
                "max_prefill_tokens": scheduler.max_prefill_tokens,
            },
        }
        (self.directory / "pd-domain.json").write_text(json.dumps(domain) + "\n", encoding="utf-8")
        self.domain_written = True

    def record(self, name, rows):
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    def transfer(self, original, queue, *args, **kwargs):
        requests = original(queue, *args, **kwargs)
        observed = []
        for request in requests:
            if request.rid in self.requests:
                observed.append({
                    "request": request.rid, "slot": request.req_pool_idx, "committed": request.kv_committed_len,
                    "at_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
                })
        if observed:
            self.record("pd-transfers.jsonl", observed)
        return requests

    def admit(self, original, scheduler):
        before = [request.rid for request in scheduler.waiting_queue]
        if self.requests.isdisjoint(before):
            return original(scheduler)
        if not self.domain_written:
            self.record_domain(scheduler)

        self.prepared = None
        active = scheduler.running_batch.batch_size()
        limit = min(scheduler.max_running_requests, scheduler.req_to_token_pool.size)
        result = original(scheduler)
        if self.prepared is not None:
            self.prepared.update({
                "admitted": [request.rid for request in result.reqs],
                "waiting_before": before,
                "waiting_after": [request.rid for request in scheduler.waiting_queue],
                "active_before": active, "limit": limit,
            })
            self.record("pd-preparation.jsonl", [self.prepared])
        return result

    def prepare(self, original, batch):
        identities = [request.rid for request in batch.reqs]
        if self.requests.isdisjoint(identities):
            return original(batch)

        members = [{
            "request": request.rid, "slot": request.req_pool_idx, "committed": request.kv_committed_len,
            "prefix": len(request.prefix_indices), "length": request.extend_range.length,
        } for request in batch.reqs]
        copies = PageCopies(batch, self.markers, not self.detail.isdisjoint(identities))
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        with copies:
            result = original(batch)
        self.prepared = {
            "begin_ns": began, "end_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW), "members": members,
            "copies": copies.records, "output_size": batch.out_cache_loc.numel(),
            "writes_output": all(target is batch.out_cache_loc for target in copies.targets),
            "computed": [request.already_computed for request in batch.reqs],
            "sequence_lengths": batch.seq_lens_cpu.tolist(),
        }
        return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        hooks = (
            ("sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred", self.transfer),
            ("sglang.srt.managers.scheduler.Scheduler.get_new_prebuilt_batch", self.admit),
            ("sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_prebuilt", self.prepare),
        )
        for name, operation in hooks:
            HookRegistry.register(name, operation, HookType.AROUND)
