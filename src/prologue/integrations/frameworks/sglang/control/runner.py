"""Initialize native scheduler metadata without loading weights or allocating KV tensors."""

from inspect import signature

import torch
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.layers.dp_attention import get_attention_tp_group, initialize_dp_attention
from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_loader.utils import get_model_architecture

from prologue.integrations.frameworks.sglang.control.resources import PageView, SlotView


class DeferredForward:
    def __init__(self, forward):
        native = signature(forward)
        self.__signature__ = native.replace(parameters=list(native.parameters.values())[1:])

    def __call__(self, *args, **kwargs):
        raise RuntimeError("native control must submit model work to Prologue")


class MetadataRunner(ModelRunner):
    def __init__(self, *args, peer, rendezvous, **kwargs):
        self.peer, self.rendezvous = peer, rendezvous
        super().__init__(*args, **kwargs)

    def load_model(self):
        architecture, _ = get_model_architecture(self.model_config)
        self.model = torch.nn.Module()
        self.model.forward = DeferredForward(architecture.forward)
        self.model.start_layer, self.model.end_layer = get_pp_indices(
            self.model_config.num_hidden_layers, self.pp_rank, self.server_args.pp_size,
        )
        self.dtype = self.model_config.dtype
        self.sliding_window_size = self.model_config.sliding_window_size
        self.weight_load_mem_usage = 0
        self.load_config = None

    def init_threads_binding(self):
        torch.set_num_threads(1)

    def init_torch_distributed(self):
        parallel_state.init_distributed_environment(
            backend="gloo", world_size=self.server_args.pp_size, rank=self.pp_rank, local_rank=0,
            distributed_init_method=self.rendezvous,
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=self.server_args.pp_size,
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
