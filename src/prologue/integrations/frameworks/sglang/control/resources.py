"""Expose core resource bindings as native CPU metadata views."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from prologue.integrations.frameworks.sglang.radix import RadixPages


class SlotView:
    def __init__(self, peer, size, context, pre_alloc_size=0):
        self.peer, self.size, self.max_context_len = peer, size, context
        self.pre_alloc_size = pre_alloc_size
        self.device = "cpu"
        self.req_to_token = torch.zeros((size + pre_alloc_size + 1, context), dtype=torch.int32)

    def available_size(self):
        return self.peer.call("capacity")["slots"]

    @property
    def free_slots(self):
        return [slot + 1 for slot in self.peer.call("free_slots")]

    def alloc(self, requests):
        slots = self.peer.call("bind_slots", requests=[request.rid for request in requests])
        for request, slot in zip(requests, slots, strict=True):
            request.req_pool_idx = slot + 1
        return [request.req_pool_idx for request in requests]

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def free(self, request):
        self.peer.call("release_slot", request=request.rid)
        request.req_pool_idx = None


class PageView:
    # Keep SGLang's batching of physical returns at its original call boundary.
    free_group_begin = BaseTokenToKVPoolAllocator.free_group_begin
    free_group_end = BaseTokenToKVPoolAllocator.free_group_end

    def __init__(self, peer, tokens, page_size, storage=None):
        self.peer, self.size, self.page_size = peer, tokens, page_size
        self.device = "cpu"
        self.storage = storage if storage is not None else SimpleNamespace(size=tokens, page_size=page_size)
        self.is_not_in_free_group, self.free_group = True, []
        self.members = ()
        self.batch = None

    def get_kvcache(self):
        return self.storage

    def available_size(self):
        return self.peer.call("capacity")["kv_tokens"]

    def reserve(self, starts, ends, use_prefix=False):
        demand = []
        for request, begin, end in zip(self.members, starts, ends, strict=True):
            row = {"request": request.rid, "begin": begin, "end": end}
            if use_prefix and request.cache_protected_len:
                row["prefix_pages"] = RadixPages.indices(request.prefix_indices[:request.cache_protected_len],
                                                         self.page_size)
            demand.append(row)
        indices = self.peer.call("grow", members=demand)

        # SGLang reserves the first physical page for dummy rows.
        return torch.tensor(indices, dtype=torch.int32) + self.page_size

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                     last_loc, extend_num_tokens, num_new_pages=None):
        return self.reserve(prefix_lens_cpu.tolist(), seq_lens_cpu.tolist(), use_prefix=True)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        ends = seq_lens_cpu.tolist()
        return self.reserve([end - 1 for end in ends], ends)

    def alloc(self, count):
        if self.batch is None:
            member, = self.members
            end = member.kv_allocated_len
            return self.reserve([end - count], [end])

        # The scalar allocator receives one count for the entire batch, before request offsets are updated.
        lengths = self.batch.seq_lens_cpu.tolist()
        extend = self.batch.forward_mode.is_extend()
        starts, ends = (self.batch.prefix_lens, lengths) if extend else (lengths, [length + 1 for length in lengths])
        if sum(end - start for start, end in zip(starts, ends, strict=True)) != count:
            raise ValueError("scalar KV allocation disagrees with the native batch extents")
        return self.reserve(starts, ends, use_prefix=extend)

    def free(self, indices):
        if not self.is_not_in_free_group:
            self.free_group.append(indices)
        elif indices.numel():
            pages = torch.unique(indices // self.page_size - 1).tolist()
            self.peer.call("release_pages", pages=pages)

    @contextmanager
    def binding(self, members, batch=None):
        previous = self.members, self.batch
        self.members, self.batch = members, batch
        try:
            yield
        finally:
            self.members, self.batch = previous

    @staticmethod
    def bind_members(original, batch, *args, **kwargs):
        with batch.token_to_kv_pool_allocator.binding(batch.reqs, batch):
            return original(batch, *args, **kwargs)

    @classmethod
    def install(cls):
        for method in ("alloc_for_extend", "alloc_for_decode"):
            HookRegistry.register("sglang.srt.mem_cache.common." + method, cls.bind_members, HookType.AROUND)
        HookRegistry.apply_hooks()
