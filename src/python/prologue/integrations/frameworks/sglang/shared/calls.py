"""Segment native work at host reads, pipeline messages and collective boundaries."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.runtimes.pytorch.shared import CpuTransfer


def native_implementation(original):
    return getattr(original, "prologue_implementation", None) or f"{original.__module__}.{original.__qualname__}"


class PipelineMessages:
    def __init__(self, calls, *, wire_device=None):
        self.calls = calls
        self.wire_device = wire_device
        self.message = ContextVar("pipeline_message", default=None)

    def work(self):
        work = self.calls.work(None)
        work["context"] = {
            "pipeline": PipelineABI.control_context(self.calls),
            "message": None if self.message.get() is None else dict(self.message.get()),
        }
        return work

    def signal_work(self, stream):
        work = self.work()
        work["context"]["record_stream"] = PipelineABI.stream_role(self.calls.scheduler, stream)
        return work

    def wait_work(self, producer, *, device, stream=None):
        work = self.work()

        # Waiting consumes the recorded producer; the current scheduler state locates this occurrence.
        work["observation_context"] = work["context"]
        work["context"] = {"producer": producer, "device_wait": device}
        if device:
            work["context"]["wait_stream"] = PipelineABI.stream_role(self.calls.scheduler, stream)
        return work

    def dictionary(self, direction, original, group, *args, **kwargs):
        if not self.calls.enabled or group.world_size == 1:
            return original(group, *args, **kwargs)
        previous = self.message.set({"direction": direction, "kind": None, "field": None})
        try:
            return original(group, *args, **kwargs)
        finally:
            self.message.reset(previous)

    def split(self, original, tensor_dict):
        metadata, tensors = original(tensor_dict)
        message = self.message.get()
        if message is None or message["direction"] != "send":
            return metadata, tensors

        TensorMetadata = resolve(symbols().tensor_metadata)

        message["kind"] = tensor_dict.get("__msg_type__")
        names = [name for name, value in metadata if isinstance(value, TensorMetadata)]
        if self.wire_device is not None:
            metadata = self.devices(metadata, "meta", self.wire_device)
        return metadata, self.fields(zip(names, tensors, strict=True), paired=False)

    def receive_metadata(self, original, group, *args, **kwargs):
        metadata = original(group, *args, **kwargs)
        message = self.message.get()
        if message is None or message["direction"] != "receive":
            return metadata
        message["kind"] = dict(metadata).get("__msg_type__")
        if self.wire_device is not None:
            metadata = self.devices(metadata, self.wire_device, "meta")
        return self.fields(metadata, paired=True)

    @staticmethod
    def devices(metadata, source, target):
        TensorMetadata = resolve(symbols().tensor_metadata)

        translated = []
        for name, value in metadata:
            if isinstance(value, TensorMetadata) and value.device == source:

                # Native device.type gives each descriptor its own string for pickle's memo table.
                value = value._replace(device=target.encode().decode())
            translated.append((name, value))
        return translated

    def fields(self, entries, *, paired):
        # Attribute the native loop's current field, including equal-shaped tensors and sliced views.
        for name, value in entries:
            self.message.get()["field"] = name
            yield (name, value) if paired else value

    def install(self):
        module = symbols().parallel_state + "."
        for target, hook in (
            (module + "GroupCoordinator.send_tensor_dict", partial(self.dictionary, "send")),
            (module + "GroupCoordinator.recv_tensor_dict", partial(self.dictionary, "receive")),
            (module + "_split_tensor_dict", self.split),
            (module + "GroupCoordinator.recv_object", self.receive_metadata),
        ):
            HookTable.around(target, hook)


class PipelineABI:
    stream_roles = ("schedule", "forward", "copy")

    @staticmethod
    def stream_declarations(recording):
        declarations, waits = [], {}
        for scope in recording["scopes"]:
            context = scope["work"].get("context", {})
            if scope["role"] == "signal_record":
                port, = (port for port in scope["ports"] if port["use"] == "signal" and port["direction"] == "output")
                declarations.append((port["point"], context["record_stream"]))
            elif scope["role"] == "signal_wait":
                port, = (port for port in scope["ports"] if port["use"] == "signal" and port["direction"] == "input")
                declarations.append((port["point"], context["producer"]["record_stream"]))
                if context["device_wait"]:
                    declarations.append((scope["device_entry"], context["wait_stream"]))
            elif scope["role"] == "stream_wait":
                waits[scope["id"]] = context["streams"]
                declarations.append((scope["device_entry"], context["streams"]["target"]))

        for query in recording["state_queries"]:
            if query["kind"] == "event":
                context = recording["scopes"][query["scope"]]["work"]["context"]
                declarations.append((query["point"], context["producer"]["record_stream"]))

        # Declarations name native streams; runtime evidence supplies their actual identities.
        points = {}
        for point, role in declarations:
            if role not in PipelineABI.stream_roles or points.setdefault(point, role) != role:
                raise ValueError("native event has an invalid or conflicting logical stream role")
        if any(role not in PipelineABI.stream_roles for streams in waits.values() for role in streams.values()):
            raise ValueError("native stream wait has an unknown logical stream role")
        return points, waits

    @staticmethod
    def pending_model_sends(scheduler):
        # Empty request forwarding continues while idle; proxy/output sends still retain model work.
        return scheduler.ps.pp_size > 1 and bool(scheduler.send_proxy_work or scheduler.send_output_work)

    @staticmethod
    def control_context(calls):
        scheduler = calls.scheduler

        # Fixed native slots identify placement, not polling rounds; absent and empty batches remain distinct.
        return {
            "current": None if scheduler.cur_batch is None else calls.progress(scheduler.cur_batch.reqs),
            "slots": [None if batch is None else calls.progress(batch.reqs) for batch in scheduler.mbs],
            "running_slots": [
                index for index, batch in enumerate(scheduler.running_mbs) if batch is scheduler.running_batch
            ],
        }

    @staticmethod
    def proxy_tensors(proxy):
        import torch

        # Native PP dictionaries also retain routing metadata such as the message kind.
        return {name: value for name, value in proxy.tensors.items() if isinstance(value, torch.Tensor)}

    @staticmethod
    def tensor_parameters(tensor):
        return {f"extent_{axis}": size for axis, size in enumerate(tensor.shape)}

    @staticmethod
    def gather_condition(tensor, dim, group):
        return {"dtype": str(tensor.dtype), "dim": dim % tensor.ndim, "group": group}

    @staticmethod
    def streams(scheduler, target, source):
        return {
            direction: PipelineABI.stream_role(scheduler, stream)
            for direction, stream in (("source", source), ("target", target))
        }

    @staticmethod
    def stream_role(scheduler, stream):
        stream = scheduler.device_module.current_stream() if stream is None else stream
        return next(role for role in PipelineABI.stream_roles if stream == getattr(scheduler, f"{role}_stream"))


@dataclass
class ActionCall:
    operation: str
    parameters: dict[str, int]
    work: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    condition: dict[str, Any] = field(default_factory=dict)
    guards: tuple = ()
    input_use: str = "device"
    materializer: str | None = None
    implementation: str | None = None


@dataclass
class CallFrame:
    operation: str
    original: Any
    parameters: dict[str, int]
    work: dict[str, Any]
    inputs: Any = None
    guards: tuple = ()
    host_only: bool = False
    outputs: list[dict[str, Any]] = field(default_factory=list)
    entry: str = "call.enter"
    exit: str = ""
    entry_context: dict = field(default_factory=dict)
    exit_context: dict = field(default_factory=dict)
    segment: Any = None
    values: dict = field(default_factory=dict)
    observations: list[int] = field(default_factory=list)
    tensor_inputs: dict = field(default_factory=dict)
    metadata_inputs: dict = field(default_factory=dict)
    tensor_allocations: dict = field(default_factory=dict)
    tensor_outputs: dict = field(default_factory=dict)
    condition: dict = field(default_factory=dict)
    segment_facts: dict = field(default_factory=dict)
    host_inputs: dict = field(default_factory=dict)
    local_copies: dict = field(default_factory=dict)
    local_upload: str | None = None
    device_tail: bool = False

    @property
    def terminal(self):
        return self.exit == "call.return"

    @property
    def host_fragment(self):
        return self.host_only or self.device_tail and not self.terminal

    @property
    def observed_condition(self):
        return self.condition | self.segment_facts

    @property
    def continuation(self):
        return {
            "owner": self.operation, "entry": self.entry, "exit": self.exit,
            "entry_context": self.entry_context, "exit_context": self.exit_context,
        }

    def child(self, work):
        # Preserve the caller's logical position, not a run-local call counter or resource binding.
        if "observation_context" in self.work:
            work.setdefault("observation_context", self.work["observation_context"])
        work["callers"] = [*self.work.get("callers", ()), {
            "owner": self.operation, "context": self.work.get("context", {}),
            "entry": self.entry, "entry_context": self.entry_context,
        }]


class CallSegments:
    def __init__(self) -> None:
        self.current = ContextVar("native_call_segments", default=None)
        self.reading = ContextVar("native_state_read", default=None)
        self.communicating = ContextVar("native_cpu_protocol", default=None)

    @property
    def protocol_enabled(self):
        return self.enabled

    def enclosing(self, frame):
        return nullcontext()

    def fragment(self, frame):
        raise NotImplementedError

    def request_id(self, native_id):
        raise NotImplementedError

    def read_boundary(self, kind, original, work, condition):
        raise NotImplementedError

    def device_collective(self, original, arguments):
        return original(**arguments)

    @contextmanager
    def read(self, kind, original, work, *, read=None):
        work = dict(work)
        work["context"] = {**work["context"], "query": kind}
        condition = {"query": kind}
        if read is not None:
            work["context"]["read"] = read
            condition["read"] = read

        # Finish preceding CPU work before observing state; do not add a wait for the queried producer.
        with self.separate("query", work), self.read_boundary(kind, original, work, condition) as query:
            query["clocks"] = []
            previous = self.reading.set(query)
            try:
                yield query
            finally:
                self.reading.reset(previous)

    def clock_sample(self, original, query):
        value = original()
        return value, round(value * 1_000_000_000)

    def clock_read(self, original, description):
        if not self.enabled:
            return self.clock_sample(original, None)[0]
        reading = self.reading.get()
        if reading is not None:

            # A poll's internal clock belongs to that read, not to a second independently expanded query.
            value, ticks = self.clock_sample(original, reading)
            reading["clocks"].append({"read": description, "result": ticks})
            return value
        parent = self.current.get()
        work = self.scheduler_work() if parent is None else dict(parent.work)
        with self.read("clock", original, work, read=description) as query:
            value, query["result"] = self.clock_sample(original, query)
        return value

    def work(self, batch, members=None, *, phase="prefill"):
        work = {} if batch is None else {"context": {"members": self.progress(batch.reqs)}}
        if members is not None:
            work.update(
                phase=phase, rank=self.rank,
                members=[{**member, "id": self.request_id(member["id"])} for member in members],
            )
        return work

    def progress(self, requests):
        # Logical request progress is comparable across runs; native slots and object handles are not.
        return [
            {
                "id": self.request_id(req.rid), "generated_tokens": len(req.output_ids),
                "committed_tokens": req.kv_committed_len,
                "extend": None if req.extend_range is None else [req.extend_range.start, req.extend_range.end],
            }
            for req in requests
        ]

    def request_work(self, request, begin=0, end=None):
        end = len(request.origin_input_ids) if end is None else end
        work = self.work(None, [{"id": request.rid, "context_tokens": begin, "new_tokens": end - begin}])

        # Token coverage describes the graph's accesses; request progress identifies this native invocation.
        work["context"] = {"request": self.progress((request,))[0]}
        return work

    def scheduler_work(self):
        scheduler = self.scheduler
        queues = {
            "waiting": scheduler.waiting_queue,
            "running": scheduler.running_batch.reqs,
            "previous": scheduler.last_batch.reqs if scheduler.last_batch is not None else (),
            "chunked": (scheduler.chunked_req,) if scheduler.chunked_req is not None else (),
        }
        mode = scheduler.server_args.disaggregation_mode
        if mode == "prefill":
            queues.update(
                bootstrap=scheduler.disagg_prefill_bootstrap_queue.queue,
                transferring=scheduler.disagg_prefill_inflight_queue,
            )
        elif mode == "decode":
            prealloc = scheduler.disagg_decode_prealloc_queue
            queues.update(
                prealloc=[entry.req for entry in prealloc.queue],
                pending=[entry.req for entry in prealloc.pending_reqs],
                retracted=prealloc.retracted_queue,
                transferring=[entry.req for entry in scheduler.disagg_decode_transfer_queue.queue],
            )

        # Snapshot native membership and progress, without inventing a unique identity for repeated polls.
        work = self.work(None)
        work["context"] = {name: self.progress(requests) for name, requests in queues.items()}
        if scheduler.ps.pp_size > 1:
            work["context"]["pipeline"] = PipelineABI.control_context(self)
        return work

    def resume(self, frame, entry="call.enter", context=None, tensor_inputs=None):
        frame.entry = entry
        frame.entry_context = {} if context is None else context
        frame.exit = ""
        frame.exit_context = {}
        frame.segment_facts.clear()
        if entry != "call.enter" and tensor_inputs is not None:

            # A resource-only child leaves borrowed inputs live; explicit returns replace the continuation contract.
            frame.tensor_inputs = dict(tensor_inputs)
        frame.tensor_outputs = {}
        frame.segment = self.fragment(frame)
        frame.segment.__enter__()

    def suspend(self, frame, boundary, context=None, error=None):
        segment, frame.segment = frame.segment, None
        if segment is None:
            return
        frame.exit = boundary
        frame.exit_context = {} if context is None else context
        segment.__exit__(
            type(error) if error is not None else None, error, error.__traceback__ if error is not None else None
        )

    @contextmanager
    def separate(self, operation, work):
        parent = self.current.get()
        context = work.get("context", {})
        if parent is not None:
            parent.child(work)
            self.suspend(parent, operation, context)
        previous = self.current.set(None)
        try:
            yield
        finally:
            self.current.reset(previous)

        # Independently submitted actions must not also appear inside their caller's segment.
        if parent is not None:
            self.resume(parent, operation, context)

    @contextmanager
    def call(
        self, operation, original, *, parameters=None, work=None, inputs=None, guards=(),
        host_only=False, device_tail=False, tensor_inputs=None, tensor_outputs=None, metadata_inputs=None,
        allocations=None, condition=None, host_inputs=None, local_copies=None, local_upload=None,
    ):
        parent = self.current.get()
        work = dict(parent.work if work is None else work)
        context = work.get("context", {})
        if parent is not None:
            parent.child(work)
            self.suspend(parent, operation, context)

        # Continuations inherit their owner's inputs; independent resource APIs supply their own explicit inputs.
        frame = CallFrame(
            operation, original,
            dict(parent.parameters if parameters is None else parameters),
            work, inputs, guards, host_only,
            tensor_inputs={} if tensor_inputs is None else tensor_inputs,
            metadata_inputs={} if metadata_inputs is None else metadata_inputs,
            tensor_allocations={} if allocations is None else allocations,
            condition={} if condition is None else condition,
            host_inputs={} if host_inputs is None else host_inputs,
            local_copies={} if local_copies is None else local_copies,
            local_upload=local_upload,
            device_tail=device_tail,
        )
        previous = self.current.set(frame)
        try:
            with self.enclosing(frame):
                self.resume(frame)
                try:
                    yield frame
                except BaseException as error:
                    self.suspend(frame, "call.error", error=error)
                    raise
                else:
                    self.suspend(frame, "call.return")
        finally:
            self.current.reset(previous)

        # Resume only after a successful child return; a failing call cannot publish another successful segment.
        if parent is not None:
            self.resume(parent, operation, context, tensor_outputs)


class HostReads:
    method = "torch.Tensor.tolist"

    def __init__(self, calls) -> None:
        self.calls = calls

    @staticmethod
    def result_inputs(result):
        tensor = result.next_token_ids
        return {} if tensor is None else {"next_token_ids": (tensor, ["members"])}

    def invoke(self, original, tensor, *args, **kwargs):
        calls = self.calls
        frame = calls.current.get()
        if not calls.enabled or frame is None:
            return original(tensor, *args, **kwargs)
        for name, (value, shape) in frame.host_inputs.items():
            if value is not tensor:
                continue
            work = {**frame.work, "context": {**frame.work["context"], "host_input": name}}
            boundary = ActionCall(
                "host_read",
                dict(frame.parameters),
                work,
                inputs={name: (tensor, shape)},
                condition={"input": name, "method": "tolist"},
                input_use="host",
                materializer=self.method,

                # This adapter-owned call has a published identity independent of its Python package layout.
                implementation="prologue.integrations.frameworks.sglang.host_reads.HostReads.invoke",
            )

            # Only this actual read owns readiness and waiting; unrelated tolist calls remain native.
            with calls.separate(boundary.operation, work), calls.boundary(self.invoke, boundary):
                return original(tensor, *args, **kwargs)
        return original(tensor, *args, **kwargs)

    def install(self):
        HookTable.around(self.method, self.invoke)


class MetadataCollectives:
    def __init__(self, calls) -> None:
        self.calls = calls

    @staticmethod
    def snapshot(arguments, root):
        import torch.distributed as distributed

        if "input_tensor" in arguments:
            inputs, outputs = arguments["input_tensor"], arguments["output_tensor"]
        else:
            inputs = outputs = arguments["tensor"]

        # A broadcast receiver's pre-call buffer is uninitialized storage, not an operand.
        values = (
            [] if root is not None and distributed.get_rank(arguments["group"]) != root else inputs.flatten().tolist()
        )
        return {"input": values}, outputs

    def invoke(self, name, signature, original, *args, **kwargs):
        calls = self.calls
        parent = calls.current.get()
        if not calls.enabled or parent is None:
            return original(*args, **kwargs)

        import torch.distributed as distributed

        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        tensor = values["input_tensor" if name == "all_gather_into_tensor" else "tensor"]
        group = distributed.group.WORLD if values["group"] is None else values["group"]
        if not calls.groups.is_cpu(group):
            return calls.device_collective(original, values)

        # Empty Gloo reductions and singleton groups do not impose a peer rendezvous.
        if distributed.get_world_size(group) == 1 or (name == "all_reduce" and tensor.numel() == 0):
            return original(*args, **kwargs)
        if values["async_op"]:
            raise ValueError("CPU metadata collectives require blocking completion")

        members = distributed.get_process_group_ranks(group)
        root = None
        if name == "broadcast":
            source = values.get("group_src")
            root = members.index(values["src"]) if source is None else source
        collective = {
            "group": calls.groups.name(group), "operation": name, "dtype": str(tensor.dtype), "root": root,
        }
        if name == "all_reduce":
            collective["reduction"] = str(values["op"])
        work = {**parent.work, "context": {**parent.work.get("context", {}), "collective": collective}}
        boundary = ActionCall(
            "metadata_collective", {"elements": tensor.numel()}, work, condition={"collective": collective},
        )

        # Submit preceding CPU work before rendezvous; the native call remains the sole owner of metadata values.
        with calls.separate(boundary.operation, work):
            return calls.collective(original, boundary, values)

    def install(self):
        import torch.distributed as distributed


        for name in ("broadcast", "all_gather_into_tensor", "all_reduce"):
            hook = partial(self.invoke, name, inspect.signature(getattr(distributed, name)))
            HookTable.around(f"torch.distributed.{name}", hook)


class MetadataWork:
    def __init__(self, owner, native, operation, transfer, group) -> None:
        self.owner, self.native = owner, native
        self.operation, self.transfer, self.group = operation, transfer, group

    def wait(self, *args, **kwargs):
        with self.owner.boundary(self.operation + "_wait", self.transfer, self.group):
            return self.native.wait(*args, **kwargs)


class MetadataTransfers:
    def __init__(self, calls, protocol=None) -> None:
        self.calls = calls
        self.protocol = protocol
        self.transfer = CpuTransfer()
        self.in_transfer = ContextVar("native_metadata_transfer", default=False)

    @contextmanager
    def boundary(self, operation, transfer, group):
        calls = self.calls
        active = calls.protocol_enabled
        parent = calls.current.get()
        work = {}
        if active:
            work = calls.scheduler_work() if parent is None else dict(parent.work)
            transfer = {**transfer, "group": calls.groups.name(group)}
            work["context"] = {**work["context"], "cpu_transfer": CpuTransfer.identity(transfer)}

        # Runtime services own protocol time; the caller must neither overlap it nor expand it as template work.
        with calls.separate("cpu_" + operation, work):
            previous = self.in_transfer.set(True)
            context = calls.communicating.set({"operation": operation, "in_flow": active, "work": work})
            try:
                yield
            finally:
                calls.communicating.reset(context)
                self.in_transfer.reset(previous)

    def invoke(self, name, signature, original, *args, **kwargs):

        # Blocking send/recv can call their async counterpart. Observe the outer native call exactly once.
        if self.in_transfer.get():
            return original(*args, **kwargs)
        native = self.transfer.native(name, signature, args, kwargs)
        if native is None:
            return original(*args, **kwargs)
        transfer, group = native
        run = original if self.protocol is None else partial(self.protocol.invoke, name, transfer, group, original)
        with self.boundary(name, transfer, group):
            result = run(*args, **kwargs)
        return MetadataWork(self, result, name, transfer, group) if name in {"isend", "irecv"} else result

    def install(self):
        for path, hook in CpuTransfer.hooks(self.invoke):
            HookTable.around(path, hook)
