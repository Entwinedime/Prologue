"""Run distributed model shards on one device and exchange their real operands through Gloo."""

import fcntl
import inspect
import os
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as distributed

from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols


class StagedWork:
    """Keep posted operands alive and publish the received values at the native wait."""

    def __init__(self, transport, native, source, copies, observation=None):
        self.transport, self.native = transport, native
        self.source, self.copies, self.observation = source, copies, observation

    def wait(self, *args, **kwargs):
        if self.native is None:
            return True
        with self.transport.auxiliary():
            with self.transport.peers():
                completed = self.native.wait(*args, **kwargs)
            if not completed:
                return completed
            for target, buffer in self.copies:
                self.transport.compute(target.copy_, buffer)

        if self.observation is not None:
            self.observation.close()
        self.native = self.source = self.copies = self.observation = None
        return completed


class ShardTransport:
    @staticmethod
    def world(size, rank, rendezvous):
        parallel_state = resolve(symbols().parallel_state)

        # SGLang reuses this host group for real logical shards placed together on device zero.
        distributed.init_process_group("gloo", init_method=rendezvous, world_size=size, rank=rank)
        parallel_state.init_distributed_environment(size, rank, rendezvous, local_rank=0, backend="gloo")

    def __init__(self, directory, auxiliary, record=None):
        self.device = Path(directory) / ".shard-device.lock"
        self.auxiliary = auxiliary
        self.record = record
        self.lease = None

    def acquire(self, handle):
        with self.auxiliary():
            fcntl.flock(handle, fcntl.LOCK_EX)
        self.lease = handle

    def release(self):
        with self.auxiliary():

            # All streams from this segment finish before a different worker can use the card.
            device_runtime().api.synchronize()
            fcntl.flock(self.lease, fcntl.LOCK_UN)
        self.lease = None

    def compute(self, original, *args, **kwargs):
        if self.lease is not None:
            return original(*args, **kwargs)
        with self.device.open("a") as handle:
            self.acquire(handle)
            try:
                return original(*args, **kwargs)
            finally:
                self.release()

    @contextmanager
    def peers(self):
        handle = self.lease
        if handle is None:
            yield
            return
        self.release()
        try:
            yield
        finally:
            self.acquire(handle)

    def exchange(self, name, signature, source_name, output_name, original, *args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        values = arguments.arguments
        source = values[source_name]
        if source.device.type == "cpu":
            with self.peers():
                return original(*args, **kwargs)
        output = values[output_name]
        outputs = output if isinstance(output, list) else [output]
        with ExitStack() as observation:
            if self.record is not None:
                observation.enter_context(self.record(name, source, outputs))
            with self.auxiliary():
                host = self.compute(source.detach().cpu)
                values[source_name] = host
                if source_name == output_name:
                    buffers = [host]
                else:
                    buffers = [torch.empty_like(tensor, device="cpu") for tensor in outputs]
                    values[output_name] = buffers if isinstance(output, list) else buffers[0]

                # Gloo uses the native group, reduction or row splits on the original contributions.
                with self.peers():
                    result = original(*arguments.args, **arguments.kwargs)
                copies = tuple(zip(outputs, buffers, strict=True))
                if values.get("async_op", False):
                    return StagedWork(self, result, host, copies, observation.pop_all())
                for target, prepared in copies:
                    self.compute(target.copy_, prepared)
        return result

    def post(self, name, signature, original, *args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        tensor = arguments.arguments["tensor"]
        if tensor.device.type == "cpu":
            with self.peers():
                return original(*args, **kwargs)

        # Sends return before their peers receive. A receive restores the original tensor at Work.wait.
        receives = name == "irecv"
        measured = self.record(name, tensor, ()) if self.record is not None else nullcontext()
        with measured, self.auxiliary():
            if receives:
                buffer = torch.empty_like(tensor, device="cpu", memory_format=torch.contiguous_format)
            else:
                buffer = self.compute(tensor.detach().to, device="cpu", memory_format=torch.contiguous_format)
            arguments.arguments["tensor"] = buffer
            with self.peers():
                work = original(*arguments.args, **arguments.kwargs)
        return StagedWork(self, work, buffer, ((tensor, buffer),) if receives else ())

    @staticmethod
    def expert_buffer(buffer, group, num_nvl_bytes=0, num_rdma_bytes=0, low_latency_mode=False, **options):
        if low_latency_mode or os.environ.get("DEEP_USE_ALLTOALL_MODE") != "1":
            raise ValueError("single-device expert exchange requires the native normal AllToAll layout")

        # This native layout uses the process group and per-dispatch row splits. Its device buffer belongs to other
        # transports.
        buffer.group, buffer.rank, buffer.group_size = group, group.rank(), group.size()
        buffer.low_latency_mode, buffer.alltoall_mode = False, True
        buffer._alltoall_layout = None

    def install_collectives(self, ep_size):
        for name, source, output in (
            ("all_reduce", "tensor", "tensor"), ("broadcast", "tensor", "tensor"),
            ("all_gather_into_tensor", "input_tensor", "output_tensor"), ("all_gather", "tensor", "tensor_list"),
            ("reduce_scatter_tensor", "input", "output"), ("all_to_all_single", "input", "output"),
        ):
            signature = inspect.signature(getattr(distributed, name))
            HookTable.around(f"torch.distributed.{name}", partial(self.exchange, name, signature, source, output))

        for name in ("isend", "irecv"):
            signature = inspect.signature(getattr(distributed, name))
            HookTable.around(f"torch.distributed.{name}", partial(self.post, name, signature))

        if ep_size > 1:
            HookTable.replace("deep_ep.Buffer.__init__", self.expert_buffer)

    def install(self, description):
        from prologue.integrations.frameworks.sglang.observation.local.shard_weights import ShardWeights

        self.install_collectives(description["ep_size"])
        weights = ShardWeights(Path(description["instance_directory"]), description["worker_count"], self.auxiliary)
        HookTable.around(symbols().model_runner_init_torch_distributed, weights.measure)

        # Park loaded layers before granting the device to another loader or entering the model runner's cross-rank
        # barrier.
        for target in (
            symbols().default_model_loader_load_model,
            symbols().dummy_model_loader_load_model,
        ):
            HookTable.around(target, partial(self.compute, weights.load))
        for target in (
            symbols().model_runner_forward,
            symbols().model_runner_sample,
        ):
            HookTable.around(target, self.compute)
