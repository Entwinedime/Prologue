"""Execute CPU metadata operations and keep native tensors bound to simulated producers."""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count

import torch
import torch.distributed as distributed
from torch.multiprocessing.reductions import StorageWeakRef

from prologue.integrations.runtimes.pytorch.shared import CpuTransfer


@dataclass(eq=False)
class MetadataWork:
    peer: object
    handle: object
    tensor: torch.Tensor | None
    wait_operation: str
    context: object = None
    finished: bool = field(default=False, init=False)

    def wait(self):
        if self.handle is not None and not self.finished:
            message = self.handle if self.context is None else {"key": self.handle, "execution": self.context.get()}
            if self.context is not None and message["execution"] is None:
                raise ValueError("CPU metadata wait requires its native protocol context")
            values = self.peer.exchange(self.wait_operation, message)
            if self.tensor is not None:
                self.tensor.copy_(torch.tensor(values, dtype=self.tensor.dtype).reshape_as(self.tensor))
            self.finished = True
        return True


class TensorValues:
    def __init__(self, namespace) -> None:
        self.namespace = namespace
        self.sequence = count()
        self.bindings = {}

    def allocate(self, tensor):
        storage = tensor.untyped_storage()
        entry = self.bindings.get(storage._cdata)
        if entry is None or entry[0].expired():
            binding = {"identity": f"native-storage/{self.namespace}/{next(self.sequence)}", "version": 0}
        else:
            binding = {**entry[1], "version": entry[1]["version"] + 1}
        self.bindings[storage._cdata] = StorageWeakRef(storage), binding
        return binding

    def binding(self, tensor, name):
        entry = self.bindings.get(tensor.untyped_storage()._cdata)
        if entry is not None and not entry[0].expired():
            return entry[1]
        if tensor.numel() != 0:
            raise ValueError(f"native tensor has no live core producer: {name}")

        # Empty buffers carry geometry without requiring transport or a producer.
        return self.allocate(tensor)

    def alias(self, source, target):
        self.publish({"value": target}, {"value": self.binding(source, "value")})

    def inputs(self, tensors):
        return {name: self.binding(tensor, name) for name, tensor in tensors.items()}

    def publish(self, tensors, values):
        for name, tensor in tensors.items():
            storage = tensor.untyped_storage()

            # References do not retain native allocations or make their simulated contents ready.
            self.bindings[storage._cdata] = StorageWeakRef(storage), values[name]

    def live(self):
        self.bindings = {key: entry for key, entry in self.bindings.items() if not entry[0].expired()}
        return sorted({entry[1]["identity"] for entry in self.bindings.values()})


def install_distributed(bridge, names):
    for name in names:
        operation = getattr(bridge, name)
        setattr(distributed, name, operation)
        setattr(distributed.distributed_c10d, name, operation)


class CpuCollectives:
    """Return the virtual protocol's collective result to native control at its existing collective boundary."""

    def __init__(self, peer, world) -> None:
        self.peer = peer
        self.world = world

    @classmethod
    def install(cls, peer, world) -> None:
        install_distributed(cls(peer, world), ("broadcast", "all_gather_into_tensor", "all_reduce"))

    def exchange(self, kind, tensor, output, group, async_op, **parameters):
        if tensor.device.type != "cpu" or output.device.type != "cpu":
            raise ValueError("Native metadata collectives require CPU tensors")

        # Gloo returns locally for an empty reduction; posting it would introduce a peer rendezvous.
        handle = None
        if kind != "minimum" or tensor.numel() != 0:
            group = distributed.group.WORLD if group is None else group
            members = tuple(self.world[rank] for rank in distributed.get_process_group_ranks(group))
            handle = self.peer.exchange(
                "collective_post",
                {
                    "group": group.group_name, "members": members, "kind": kind,
                    "dtype": str(tensor.dtype), "values": tensor.flatten().tolist(), **parameters,
                },
            )

        work = MetadataWork(self.peer, handle, output, "collective_wait")
        if async_op:
            return work
        work.wait()

    def broadcast(self, tensor, src=None, group=None, async_op=False, group_src=None):
        if group_src is not None:
            src = distributed.get_process_group_ranks(group)[group_src]
        return self.exchange("broadcast", tensor, tensor, group, async_op, source=self.world[src])

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        return self.exchange("gather", input_tensor, output_tensor, group, async_op)

    def all_reduce(self, tensor, op=distributed.ReduceOp.SUM, group=None, async_op=False):
        if op != distributed.ReduceOp.MIN:
            raise ValueError("Native metadata polling requires a MIN reduction")
        return self.exchange("minimum", tensor, tensor, group, async_op)


@dataclass
class CollectiveCall:
    header: dict
    values: dict = field(default_factory=dict)
    waiting: dict = field(default_factory=dict)
    consumed: set = field(default_factory=set)
    result: list | None = None


class CollectiveCalls:
    def __init__(self) -> None:
        self.pending = {}
        self.positions = {}

    def post(self, channel, rank, message) -> None:
        members = message["members"]
        if rank not in members:
            raise ValueError("Native CPU collective has an invalid participant")

        # Participants can enter the next broadcast while another receiver is still in the preceding call.
        group = members, message["group"]
        participant = group, rank
        position = self.positions.get(participant, 0)
        key = (*group, position)
        header = {name: value for name, value in message.items() if name != "values"}
        call = self.pending.setdefault(key, CollectiveCall(header))

        if call.header != header:
            raise ValueError("Native CPU collective participants disagree on the operation")
        if call.values and len(next(iter(call.values.values()))) != len(message["values"]):
            raise ValueError("Native CPU collective participants disagree on tensor extent")

        self.positions[participant] = position + 1
        call.values[rank] = message["values"]
        kind = message["kind"]
        if kind == "broadcast":
            source = message["source"]
            if source not in members:
                raise ValueError("Native CPU broadcast source is not a group member")

            # A root can enter another broadcast while late receivers still need this payload.
            if source in call.values:
                call.result = call.values[source]
        elif len(call.values) == len(members):
            if kind == "gather":
                call.result = [value for member in members for value in call.values[member]]
            elif kind == "minimum":
                dtype = getattr(torch, message["dtype"].removeprefix("torch."))
                tensors = torch.tensor([call.values[member] for member in members], dtype=dtype)
                call.result = tensors.amin(dim=0).tolist()
            else:
                raise ValueError(f"Unknown native metadata collective: {kind}")

        # Acknowledge submission independently of result readiness, so every actor can submit its graph.
        channel.send(key)
        self.complete(key, call)

    def wait(self, channel, rank, handle) -> None:
        key = tuple(handle)
        call = self.pending[key]
        if rank not in call.values or rank in call.consumed:
            raise ValueError("Native CPU collective wait has no outstanding local operation")
        call.waiting[rank] = channel
        self.complete(key, call)

    def complete(self, key, call) -> None:
        if call.result is None:
            return
        for rank, channel in call.waiting.items():
            channel.send(call.result)
            call.consumed.add(rank)
        call.waiting.clear()

        # Completion of one actor never reclaims data still owned by another actor's native Work.
        if len(call.consumed) == len(call.header["members"]):
            del self.pending[key]

    def blocked(self):
        return {channel for call in self.pending.values() for channel in call.waiting.values()}


class CpuPointToPoint(CpuTransfer):
    def __init__(self, peer, world, context) -> None:
        super().__init__(world)
        self.peer, self.context = peer, context

    def install(self) -> None:
        install_distributed(self, CpuTransfer.operations)

    def post(self, tensor, group, tag, endpoint, group_endpoint, receiving):
        if tensor.device.type != "cpu":
            raise ValueError(
                "Native metadata transport requires CPU tensors; device transfers belong to graph execution"
            )

        execution = self.context.get()
        if execution is None:
            raise ValueError("CPU metadata post requires its native protocol context")
        message = self.describe(tensor, group, tag, endpoint, group_endpoint, receiving)

        if not receiving:
            message["values"] = tensor.flatten().tolist()
        handle = self.peer.exchange("metadata_post", {**message, "execution": execution})
        return MetadataWork(self.peer, handle, tensor if receiving else None, "metadata_wait", context=self.context)

    def isend(self, tensor, dst=None, group=None, tag=0, group_dst=None):
        return self.post(tensor, group, tag, dst, group_dst, False)

    def irecv(self, tensor, src=None, group=None, tag=0, group_src=None):
        return self.post(tensor, group, tag, src, group_src, True)

    def send(self, tensor, dst=None, group=None, tag=0, group_dst=None):
        self.isend(tensor, dst, group, tag, group_dst).wait()

    def recv(self, tensor, src=None, group=None, tag=0, group_src=None):
        work = self.irecv(tensor, src, group, tag, group_src)
        work.wait()
        return self.world.index(work.handle[1])


@dataclass
class MetadataTransfer:
    values: list | None = None
    consumed: set = field(default_factory=set)


class PointToPointCalls:
    """Keep payloads and native channel positions; the core owns readiness and suspended consumers."""

    def __init__(self) -> None:
        self.positions = defaultdict(count)
        self.pending = {}

    def post(self, rank, message):
        source, target = message["source"], message["target"]
        receiving = message["receiving"]
        if source == target or rank != (target if receiving else source):
            raise ValueError("native metadata transfer has an invalid endpoint")
        route = CpuTransfer.route(message)
        key = (*route, next(self.positions[route, rank]))
        transfer = self.pending.setdefault(key, MetadataTransfer())
        if not receiving:
            transfer.values = message["values"]
        return key

    def consume(self, rank, handle):
        key = tuple(handle)
        transfer = self.pending[key]
        if rank not in key[1:3] or rank in transfer.consumed or transfer.values is None:
            raise ValueError("core released a metadata consumer without a live matching payload")
        transfer.consumed.add(rank)
        values = transfer.values if rank == key[2] else None
        if len(transfer.consumed) == 2:
            del self.pending[key]
        return values
