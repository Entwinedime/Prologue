"""Run logical native ranks on one device with host-staged transport and layer weights."""

import fcntl
import inspect
import os
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial, update_wrapper
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as distributed

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.model import ModelScopes


class LayerWeights:
    def __init__(self, tensors, auxiliary):
        self.auxiliary = auxiliary
        self.storage = [(tensor, torch.empty(tensor.untyped_storage().nbytes(), dtype=torch.uint8, device="cpu"))
                        for tensor in tensors]

    def move(self, to_device):
        with self.auxiliary():
            if not to_device:

                # Kernels on other native streams can still be reading this layer after forward returns.
                device_runtime().api.synchronize()
            for tensor, backing in self.storage:
                device_runtime().move_weight(tensor, backing, to_device)

    def forward(self, original, *args, **kwargs):
        self.move(True)
        try:
            return original(*args, **kwargs)
        finally:
            self.move(False)


class ShardWeights:
    def __init__(self, directory, workers, auxiliary):
        self.directory, self.workers, self.auxiliary = directory, workers, auxiliary

    def measure(self, original, runner, *args, **kwargs):
        available_gib = original(runner, *args, **kwargs)
        self.share = int(available_gib * (1 << 30) * runner.mem_fraction_static) // self.workers
        return available_gib

    def load(self, original, *args, **kwargs):
        model = original(*args, **kwargs)
        weights = {
            t.untyped_storage()._cdata: t
            for t in model.parameters()
            if t.device.type == platform_domain().device_type and t.numel()
        }
        footprint = sum(t.untyped_storage().nbytes() for t in weights.values())

        if footprint <= self.share:
            return model

        modules = dict(model.named_modules())
        layers = {name for name, role in ModelScopes.selected(modules).items() if role == "layer"}
        owners = defaultdict(set)
        for path, tensor in chain(
            model.named_parameters(remove_duplicate=False), model.named_buffers(remove_duplicate=False)
        ):
            container = path.rpartition(".")[0]
            while container and container not in layers:
                container = container.rpartition(".")[0]

            owners[tensor.untyped_storage()._cdata].add(container)

        # Shared embeddings, cross-layer aliases and buffers outside a layer stay available to their actual consumers.
        groups = defaultdict(list)
        for key, tensor in weights.items():
            if len(owners[key]) == 1:
                owner, = owners[key]
                if owner in layers:
                    groups[owner].append(tensor)

        device_runtime().load_weight_bridge()
        parked = 0
        for name, tensors in groups.items():
            residence = LayerWeights(tensors, self.auxiliary)
            residence.move(False)
            layer = modules[name]
            layer.forward = update_wrapper(partial(residence.forward, layer.forward), layer.forward)
            parked += sum(backing.nbytes for _, backing in residence.storage)

        write_json(
            self.directory / "weight-residency" / f"{distributed.get_rank()}.json",
            {
                "native_weight_bytes": footprint,
                "worker_share_bytes": self.share,
                "parked_bytes": parked,
                "layers": len(groups),
            },
        )

        return model


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
        self.local_only = False

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
        if self.local_only:
            raise ValueError("targeted local work cannot issue a peer transfer")
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


def run_staged_scheduler(server_args, port_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                               pp_rank, dp_rank, pipe_writer):
    run_scheduler_process = resolve(symbols().run_scheduler_process)

    device = torch.get_device_module(server_args.device)
    if device.device_count() != 1:
        raise ValueError("logical shard workers require exactly one visible accelerator")
    device.set_device(0)

    world_size = server_args.tp_size * server_args.pp_size
    rank = pp_rank * server_args.tp_size + tp_rank
    ShardTransport.world(world_size, rank, "env://")

    run_scheduler_process(server_args, port_args, 0, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                          pp_rank, dp_rank, pipe_writer)


def staged_engine():
    class StagedEngine(resolve(symbols().engine)):
        run_scheduler_process_func = staticmethod(run_staged_scheduler)

    return StagedEngine
