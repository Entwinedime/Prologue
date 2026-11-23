"""Initialize native scheduler metadata without loading weights or allocating KV tensors."""

import torch
from sglang.srt.distributed import parallel_state
from sglang.srt.layers.dp_attention import get_attention_tp_group, initialize_dp_attention
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

from prologue.integrations.frameworks.sglang.control.resources import PageView, SlotView


class MetadataRunner(ModelRunner):
    def __init__(self, *args, peer, rendezvous, **kwargs):
        self.peer, self.rendezvous = peer, rendezvous
        super().__init__(*args, **kwargs)

    def load_model(self):
        self.model = torch.nn.Module()
        self.model.start_layer = 0
        self.model.end_layer = self.model_config.num_hidden_layers
        self.dtype = self.model_config.dtype
        self.sliding_window_size = self.model_config.sliding_window_size
        self.weight_load_mem_usage = 0
        self.load_config = None

    def init_threads_binding(self):
        torch.set_num_threads(1)

    def init_torch_distributed(self):
        parallel_state.init_distributed_environment(
            backend="gloo", world_size=1, rank=0, local_rank=0, distributed_init_method=self.rendezvous,
        )
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
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
        self.req_to_token_pool = SlotView(self.peer, self.max_running_requests, context)
        self.token_to_kv_pool_allocator = PageView(self.peer, self.max_total_num_tokens, self.server_args.page_size)
        self.token_to_kv_pool = self.token_to_kv_pool_allocator.storage

    def init_shared_mooncake_transfer_engine(self):
        self.shared_mooncake_transfer_engine = None

    def init_attention_backends(self):
        self.prefill_attention_backend_str = "torch_native"
        self.decode_attention_backend_str = "torch_native"

    def init_cuda_graphs(self, capture_decode_cuda_graph=True):
        self.eager_runner = None
        self.prefill_cuda_graph_runner = None
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
