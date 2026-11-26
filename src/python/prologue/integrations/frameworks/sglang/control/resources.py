"""Expose core-owned slots, pages and cache state to the native scheduler and model runner."""

from contextlib import contextmanager

import torch

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.batches import RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout
from prologue.integrations.frameworks.sglang.shared.resources import PoolCoordinates
from prologue.integrations.frameworks.sglang.shared.selection import AllocationABI

BaseTokenToKVPoolAllocator = resolve(symbols().base_token_to_kv_pool_allocator)


class AllocationDemand:
    def __init__(self) -> None:
        for method, hook in (("alloc_for_extend", self.for_extend), ("alloc_for_decode", self.for_decode)):
            HookTable.around(symbols().common + "." + method, hook)
        HookTable.around(symbols().pre_alloc, self.preallocate)

    @staticmethod
    @contextmanager
    def demand(allocator, requests, tokens):
        allocator.demand = [
            {"request": RequestSubmissions.index(req.rid), "tokens": count}
            for req, count in zip(requests, tokens, strict=True)
        ]
        try:
            yield
        finally:
            allocator.demand = None

    def for_extend(self, original, batch):
        with self.demand(batch.token_to_kv_pool_allocator, batch.reqs, batch.extend_lens):
            return original(batch)

    def for_decode(self, original, batch, token_per_req):
        with self.demand(batch.token_to_kv_pool_allocator, batch.reqs, [token_per_req] * len(batch.reqs)):
            return original(batch, token_per_req)

    def preallocate(self, original, queue, req, prefix_indices=None, prefix_len=None, total_prefix_len=None):
        prefix_len = 0 if prefix_len is None else prefix_len
        total_prefix_len = prefix_len if total_prefix_len is None else total_prefix_len
        if total_prefix_len != prefix_len:
            raise ValueError("Decode preallocation requires resident prefixes; tiered loadback is not implemented")

        tokens = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0) - total_prefix_len
        slots = queue.req_to_token_pool

        # Native preallocation assigns req.prefix_indices after allocating; use its call argument for slot binding.
        slots.prefix_bindings[RequestSubmissions.index(req.rid)] = (
            {"node": req.last_node, "tokens": prefix_len} if prefix_len else None
        )

        try:
            with self.demand(queue.token_to_kv_pool_allocator, [req], [tokens]):
                return original(queue, req, prefix_indices, prefix_len, total_prefix_len)
        finally:
            del slots.prefix_bindings[RequestSubmissions.index(req.rid)]


class RequestSlots:
    def __init__(self, peer, size: int, context: int, extra: int) -> None:
        self.peer = peer
        self.prefix_bindings = {}
        self.size = size
        self.pre_alloc_size = extra
        self._alloc_size = size + extra + 1
        self.max_context_len = context
        self.device = "cpu"

        # This is the native tensor view of core allocation results, not an independent allocator.
        self.req_to_token = torch.zeros((self._alloc_size, context), dtype=torch.int32)

    def available_size(self):
        return self.peer.call("capacity", resource="slots")

    @property
    def free_slots(self):
        return [PoolCoordinates.native(slot) for slot in self.peer.call("free_slots")]

    def alloc(self, reqs):
        members = []
        for req in reqs:
            member = {"request": RequestSubmissions.index(req.rid)}

            # The opaque core handle supplies cached pages; any uncached tail remains request-owned.
            prefix = self.prefix_bindings.get(
                RequestSubmissions.index(req.rid),
                {"node": req.last_node, "tokens": len(req.prefix_indices)} if req.last_node is not None else None,
            )

            if prefix is not None:
                member["prefix"] = prefix
            members.append(member)

        bindings = self.peer.call("slots", members=members)
        for req, binding in zip(reqs, bindings, strict=True):
            req.req_pool_idx = PoolCoordinates.native(binding["slot"])
        return [req.req_pool_idx for req in reqs]

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def free(self, req):
        self.peer.call("release_request", request=RequestSubmissions.index(req.rid))
        req.req_pool_idx = None


class StateSlots:
    def __init__(self, size, *, peer, field):
        self.peer, self.field = peer, field
        self.size = size
        self.call("declare", capacity=size)

    def call(self, operation, **values):
        return self.peer.call("state_slots", field=self.field, operation=operation, **values)

    def available_size(self):
        return self.call("available")

    def alloc(self):
        return self.call("allocate")

    def bind(self, request, slot):
        self.call("bind", request=RequestSubmissions.index(request.rid), slot=slot)

    def free(self, slot):
        self.call("release", slot=slot)


class KvStorage:
    def __init__(self, size: int, page_size: int, layout) -> None:
        self.size = size
        self.page_size = page_size
        self.buffers = None
        if layout is not None:
            descriptor = MhaKvLayout.read(layout)
            self.start_layer, self.end_layer = descriptor.start_layer, descriptor.end_layer
            self.head_num = descriptor.head_num
            self.buffers = descriptor.buffers(size, page_size)

    def get_contiguous_buf_infos(self):
        if self.buffers is None:
            raise ValueError("PD requires an observed KV buffer layout")
        return self.buffers.native_infos()

    def maybe_get_custom_mem_pool(self):
        return None


class PageAllocator:

    # Grouping is native control logic; only the final ownership change is delegated to the core.
    free_group_begin = BaseTokenToKVPoolAllocator.free_group_begin
    free_group_end = BaseTokenToKVPoolAllocator.free_group_end

    def __init__(self, peer, size: int, page_size: int, layout, implementation: str) -> None:
        self.peer = peer
        self._prologue_allocation = AllocationABI(implementation)
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.storage = KvStorage(size, page_size, layout)
        self.is_not_in_free_group = True
        self.free_group = []
        self.demand = None

    def available_size(self):
        return self.peer.call("capacity", resource="pages") * self.page_size

    def get_kvcache(self):
        return self.storage

    def reserve(self):
        allocation = self.peer.call("allocate", members=self.demand)
        if allocation["ordering"] is not None:
            self.merge_and_sort_free(allocation["ordering"])
        return allocation["members"]

    def merge_and_sort_free(self, ordering):

        # The core already ordered the pages; ResourceCalls exposes this native operation's execution graph.
        return None

    def allocate(self, dtype):
        return self.indices(self.reserve(), dtype)

    def indices(self, bindings, dtype):

        # SGLang reserves physical page zero; the core uses zero-based page identities.
        indices = [index + self.page_size for binding in bindings for index in binding["indices"]]
        return torch.tensor(indices, dtype=dtype)

    def page_indices(self, pages):
        starts = PoolCoordinates.native(torch.tensor(pages, dtype=torch.int64)) * self.page_size
        return (starts[:, None] + torch.arange(self.page_size, dtype=torch.int64)).flatten()

    def alloc(self, need_size):
        return self.allocate(torch.int64)

    def alloc_extend(
        self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens, num_new_pages=None
    ):
        alloc_extend_naive = resolve(symbols().alloc_extend_naive)

        # The declared allocator returns int32 from its paged entry points, unlike alloc().
        bindings = self.reserve()
        result = self.indices(bindings, torch.int32)
        pages = [PoolCoordinates.native(page) for binding in bindings for page in binding["new_pages"]]
        if self._prologue_allocation.extend_branch(len(pages)) == "extend_naive":

            # Replay only index computation over core-owned pages; the helper never allocates or frees them.
            free_pages = torch.tensor(pages, dtype=torch.int64)
            alloc_extend_naive(prefix_lens, seq_lens, last_loc, free_pages, result, self.page_size, "cpu")

        return result

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        return self.allocate(torch.int32)

    def free(self, indices):
        if indices.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(indices)
            return
        pages = torch.unique(PoolCoordinates.page(indices, self.page_size)).tolist()
        destination_empty = self.peer.call("release_pages", pages=pages)
        return destination_empty, len(pages)


DecLockRefResult = resolve(symbols().dec_lock_ref_result)
EvictResult = resolve(symbols().evict_result)
IncLockRefResult = resolve(symbols().inc_lock_ref_result)
InsertResult = resolve(symbols().insert_result)
MatchResult = resolve(symbols().match_result)
RadixCache = resolve(symbols().radix_cache)
default_radix_cache_factory = resolve(symbols().default_radix_cache_factory)


class PrefixCache(RadixCache):
    @staticmethod
    def build(context, *, expected):

        # Native chunk-only storage retains its request bookkeeping and never builds a content index.
        if context.disable_radix_cache and context.effective_chunked_prefill_size is not None:
            cache = default_radix_cache_factory(context)
            native_type = type(cache)
        else:
            cache = PrefixCache(context.params)
            native_type = RadixCache

        # The resource view replaces RadixCache storage, not the source implementation or policy.
        actual = {
            "type": f"{native_type.__module__}.{native_type.__qualname__}",
            "disabled": context.disable_radix_cache,
            "eviction_policy": context.params.eviction_policy.lower(),
        }

        if actual != expected:
            raise ValueError(f"native control cache differs from template implementation: {actual} != {expected}")
        return cache

    def __init__(self, params) -> None:
        if params.is_eagle or params.eviction_policy != "lru":
            raise ValueError("core prefix resources require ordinary token keys and LRU eviction")
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.peer = self.token_to_kv_pool_allocator.peer
        self.page_size = params.page_size
        self.disable_finished_insert = params.disable_finished_insert
        self.is_eagle = False
        self.device = torch.device("cpu")

        # Native request methods remain inherited; no native tree, allocator or eviction state is created.
        self.root_node = 0
        self._reset_session_radix_state()
        self._empty_match_result = MatchResult(torch.empty(0, dtype=torch.int64), 0, 0, 0)

    def reset(self):

        # Each simulation starts with its own resource pool; native flush must not create a second tree.
        raise NotImplementedError("reset prefix state by starting a new simulation resource pool")

    def key_tokens(self, key):
        if key.extra_key is not None:
            raise ValueError("request cache namespaces are outside the declared workload content contract")
        return list(key.page_aligned(self.page_size).raw_token_ids())

    def match_prefix(self, params):
        if self.disable:
            return self._empty_match_result
        result = self.peer.call("prefix", operation="match", tokens=self.key_tokens(params.key))
        self.record_split(*result["split"])

        # The core supplies the current binding, including splits; no mirrored native tree is maintained.
        indices = self.token_to_kv_pool_allocator.page_indices(result["pages"])
        node = result["node"]
        return MatchResult(indices, node, node, node)

    def insert(self, params):
        if self.disable:
            return InsertResult(prefix_len=0)
        tokens = self.key_tokens(params.key)
        indices = params.value[: len(tokens)]
        pages = PoolCoordinates.page(indices[:: self.page_size], self.page_size).tolist()
        result = self.peer.call("prefix", operation="insert", tokens=tokens, pages=pages)
        self.record_split(*result["split"])
        return InsertResult(prefix_len=result["existing_tokens"], last_device_node=result["node"])

    def inc_lock_ref(self, node):
        delta = 0 if self.disable else self.peer.call("prefix", operation="lock", node=node, acquire=True)
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(self, node, params=None):
        delta = 0 if self.disable else self.peer.call("prefix", operation="lock", node=node, acquire=False)
        return DecLockRefResult(delta=delta)

    def evict(self, params):
        if self.disable:
            return EvictResult()
        evicted = 0
        for pages in self.peer.call("prefix", operation="evict", tokens=params.num_tokens):

            # Keep each native leaf release separate so allocator ordering remains native-visible.
            indices = self.token_to_kv_pool_allocator.page_indices(pages)
            self.token_to_kv_pool_allocator.free(indices)
            evicted += indices.numel()
        return EvictResult(num_tokens_evicted=evicted)

    def cache_unfinished_req(self, req, chunked=False):
        super().cache_unfinished_req(req, chunked=chunked)
        if not self.disable:
            self.peer.call(
                "bind_prefix",
                request=RequestSubmissions.index(req.rid),
                node=req.last_node,
                tokens=len(req.prefix_indices),
            )

    def evictable_size(self):
        return 0 if self.disable else self.peer.call("capacity", resource="evictable_tokens")

    def protected_size(self):
        return 0 if self.disable else self.peer.call("capacity", resource="protected_tokens")


parallel_state = resolve(symbols().parallel_state)
get_pp_indices = resolve(symbols().get_pp_indices)
get_attention_tp_group = resolve(symbols().get_attention_tp_group)
initialize_dp_attention = resolve(symbols().initialize_dp_attention)
get_req_to_token_extra_context_len = resolve(symbols().get_req_to_token_extra_context_len)
ModelRunner = resolve(symbols().model_runner)
MemoryPoolConfig = resolve(symbols().memory_pool_config)


class UnexecutedModel(torch.nn.Module):
    def forward(self, *args, pp_proxy_tensors=None, **kwargs):
        raise RuntimeError("Model execution belongs to the core graph")


class MetadataRunner(ModelRunner):
    def __init__(self, *args, peer, rendezvous: str, kv_layout, allocator: str, **kwargs):
        self.peer = peer
        self.rendezvous = rendezvous
        self.kv_layout = kv_layout
        self.allocator_implementation = allocator
        super().__init__(*args, **kwargs)

    def load_model(self):
        self.model = UnexecutedModel()
        self.model.start_layer, self.model.end_layer = get_pp_indices(
            self.model_config.num_hidden_layers, self.pp_rank, self.pp_size
        )
        self.dtype = self.model_config.dtype
        self.sliding_window_size = self.model_config.sliding_window_size
        self.weight_load_mem_usage = 0
        self.load_config = None

    def init_threads_binding(self):
        torch.set_num_threads(1)

    def init_torch_distributed(self):

        # Native request/control collectives run on CPU; model collectives belong to the simulated graph.
        parallel_state.init_distributed_environment(
            backend="gloo",
            world_size=self.tp_size * self.pp_size,
            rank=self.pp_rank * self.tp_size + self.tp_rank,
            local_rank=0,
            distributed_init_method=self.rendezvous,
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            expert_model_parallel_size=self.moe_ep_size,
        )
        initialize_dp_attention(self.server_args, self.model_config)
        self.tp_group = parallel_state.get_tp_group()
        self.pp_group = parallel_state.get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        return 0

    def _resolve_memory_pool_config(self, pre_model_load_memory):
        return MemoryPoolConfig(
            max_total_num_tokens=self.server_args.max_total_tokens,
            max_running_requests=self.server_args.max_running_requests,
        )

    def _init_pools(self):
        context = self.model_config.context_len + get_req_to_token_extra_context_len(self.server_args)
        extra = (
            self.server_args.disaggregation_decode_extra_slots
            if self.server_args.disaggregation_mode == "decode" else 0
        )

        # The observed worker uses paged allocation even at page_size=1; only PD defers and sorts returned pages.
        reuse = "deferred_sorted" if self.server_args.disaggregation_mode in {"prefill", "decode"} else "prepend"
        self.peer.call("declare_resources", slots=self.max_running_requests + extra,
                       page_reuse=reuse, fields=["output_token"])
        self.req_to_token_pool = RequestSlots(self.peer, self.max_running_requests, context, extra)
        self.token_to_kv_pool_allocator = PageAllocator(
            self.peer,
            self.max_total_num_tokens,
            self.server_args.page_size,
            self.kv_layout,
            self.allocator_implementation,
        )
        self.token_to_kv_pool = self.token_to_kv_pool_allocator.storage

    def init_shared_mooncake_transfer_engine(self):

        # The transfer adapter submits graph work; prediction never opens a physical transport engine.
        self.shared_mooncake_transfer_engine = None

    def init_attention_backends(self):
        self.prefill_attention_backend_str = "torch_native"
        self.decode_attention_backend_str = "torch_native"

    def init_cuda_graphs(self, capture_decode_cuda_graph=True):
        self.eager_runner = None
        self.prefill_cuda_graph_runner = None
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
