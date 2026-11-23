"""Place native parallel workers on one visible Ascend device."""

import fcntl
import inspect
import os
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
            raise ValueError("shard acquisition requires one visible Ascend device")
        torch.npu.set_device(0)

        # The native model builds its original groups and shards from this host world.
        parallel_state.init_distributed_environment(
            world_size=server_args.tp_size * server_args.pp_size,
            rank=pp_rank * server_args.tp_size + tp_rank, local_rank=0, backend="gloo",
            distributed_init_method=f"tcp://127.0.0.1:{port_args.nccl_port}",
            timeout=server_args.dist_timeout,
        )
        run_scheduler_process(server_args, port_args, 0, tp_rank, attn_cp_rank, moe_dp_rank,
                              moe_ep_rank, pp_rank, dp_rank, pipe_writer)


class StagedWork:
    def __init__(self, transport, work, source, target, output):
        self.transport = transport
        self.work = work
        self.source = source
        self.target = target
        self.output = output

    def wait(self, *args, **kwargs):
        if self.work is None:
            return True
        with self.transport.rendezvous():
            completed = self.work.wait(*args, **kwargs)
        if not completed:
            return completed

        # A receive makes the original device tensor available at the native wait.
        if self.target is not None:
            self.transport.compute(self.target.copy_, self.output)
        self.work = self.source = self.target = self.output = None
        return completed


class StagedTransport:
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
        target = values[target_name]
        host = self.compute(source.detach().cpu)
        output = host if source_name == target_name else torch.empty_like(target, device="cpu")
        values[source_name] = host
        values[target_name] = output
        with self.rendezvous():
            result = original(*arguments.args, **arguments.kwargs)
        if values.get("async_op"):
            return StagedWork(self, result, host, target, output)
        self.compute(target.copy_, output)
        return result

    def post(self, receives, signature, original, *args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        tensor = arguments.arguments["tensor"]
        if tensor.device.type == "cpu":
            with self.rendezvous():
                return original(*args, **kwargs)

        if receives:
            buffer = torch.empty_like(tensor, device="cpu", memory_format=torch.contiguous_format)
        else:
            buffer = self.compute(tensor.detach().to, device="cpu", memory_format=torch.contiguous_format)
        arguments.arguments["tensor"] = buffer
        with self.rendezvous():
            work = original(*arguments.args, **arguments.kwargs)
        return StagedWork(self, work, buffer, tensor if receives else None, buffer)

    @staticmethod
    def expert_buffer(buffer, group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False, **options):
        if low_latency_mode or os.environ.get("DEEP_USE_ALLTOALL_MODE") != "1":
            raise ValueError("single-device expert exchange requires the native normal AllToAll layout")

        # The native AllToAll layout uses a process group and real row splits without an HCCL buffer.
        buffer.group, buffer.rank, buffer.group_size = group, group.rank(), group.size()
        buffer.low_latency_mode, buffer.alltoall_mode = False, True
        buffer._alltoall_layout = None

    def install(self, pipeline, expert):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        collectives = (
            ("all_reduce", "tensor", "tensor"),
            ("broadcast", "tensor", "tensor"),
            ("all_gather_into_tensor", "input_tensor", "output_tensor"),
            ("reduce_scatter_tensor", "input", "output"),
        )
        if expert:
            collectives += (("all_to_all_single", "input", "output"),)
            HookRegistry.register("deep_ep.Buffer.__init__", self.expert_buffer, HookType.REPLACE)
        for name, source, target in collectives:
            hook = partial(self.exchange, inspect.signature(getattr(distributed, name)), source, target)
            HookRegistry.register(f"torch.distributed.{name}", hook, HookType.AROUND)
        if pipeline:
            for name in ("isend", "irecv"):
                hook = partial(self.post, name == "irecv", inspect.signature(getattr(distributed, name)))
                HookRegistry.register(f"torch.distributed.{name}", hook, HookType.AROUND)

        runner = "sglang.srt.model_executor.model_runner.ModelRunner."
        for method in ("forward", "sample"):
            HookRegistry.register(runner + method, self.compute, HookType.AROUND)
