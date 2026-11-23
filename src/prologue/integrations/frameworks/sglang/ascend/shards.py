"""Place native tensor-parallel workers on one visible Ascend device."""

import fcntl
import inspect
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as distributed
from sglang.srt.distributed import parallel_state
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.scheduler import run_scheduler_process


class SingleDeviceEngine(Engine):
    @staticmethod
    def run_scheduler_process_func(server_args, port_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank,
                                   moe_ep_rank, pp_rank, dp_rank, pipe_writer):
        if torch.npu.device_count() != 1:
            raise ValueError("tensor-parallel shard acquisition requires one visible Ascend device")
        torch.npu.set_device(0)

        # The native model builds its original groups and shards from this host world.
        parallel_state.init_distributed_environment(
            world_size=server_args.tp_size, rank=tp_rank, local_rank=0, backend="gloo",
            distributed_init_method=f"tcp://127.0.0.1:{port_args.nccl_port}",
            timeout=server_args.dist_timeout,
        )
        run_scheduler_process(server_args, port_args, 0, tp_rank, attn_cp_rank, moe_dp_rank,
                              moe_ep_rank, pp_rank, dp_rank, pipe_writer)


class StagedCollectives:
    def __init__(self, lease_path):
        self.lease_path = lease_path
        self.lease = None

    def compute(self, original, *args, **kwargs):
        if self.lease is not None:
            return original(*args, **kwargs)
        with self.lease_path.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            self.lease = lease
            try:
                return original(*args, **kwargs)
            finally:
                self.lease = None
                torch.npu.synchronize()

    @contextmanager
    def rendezvous(self):
        lease = self.lease
        if lease is None:
            yield
            return

        # A waiting worker must release the device so peers can produce their operands.
        torch.npu.synchronize()
        fcntl.flock(lease, fcntl.LOCK_UN)
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_EX)

    def exchange(self, signature, source_name, target_name, original, *args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        values = arguments.arguments
        source = values[source_name]
        if source.device.type == "cpu":
            with self.rendezvous():
                return original(*args, **kwargs)
        if values.get("async_op"):
            raise ValueError("single-device TP uses blocking tensor collectives")

        target = values[target_name]
        host = self.compute(source.detach().cpu)
        output = host if source_name == target_name else torch.empty_like(target, device="cpu")
        values[source_name] = host
        values[target_name] = output
        with self.rendezvous():
            result = original(*arguments.args, **arguments.kwargs)
        self.compute(target.copy_, output)
        return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        for name, source, target in (
            ("all_reduce", "tensor", "tensor"),
            ("broadcast", "tensor", "tensor"),
            ("all_gather_into_tensor", "input_tensor", "output_tensor"),
        ):
            hook = partial(self.exchange, inspect.signature(getattr(distributed, name)), source, target)
            HookRegistry.register(f"torch.distributed.{name}", hook, HookType.AROUND)
        runner = "sglang.srt.model_executor.model_runner.ModelRunner."
        for method in ("forward", "sample"):
            HookRegistry.register(runner + method, self.compute, HookType.AROUND)
