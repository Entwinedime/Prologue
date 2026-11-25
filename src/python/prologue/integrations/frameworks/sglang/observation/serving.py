"""Observe scheduler, page-table and disaggregated KV lifecycles."""

import inspect
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from functools import partial
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

import torch
from torch.overrides import TorchFunctionMode

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, DecisionClocks
from prologue.integrations.frameworks.sglang.shared.calls import CallSegments, HostReads, MetadataCollectives
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import DisaggregatedCalls
from prologue.integrations.frameworks.sglang.shared.resources import ResourceCalls
from prologue.integrations.frameworks.sglang.shared.selection import AllocationABI, BatchSelection
from prologue.integrations.runtimes.pytorch.capture import TensorUpload


class ReceiverTableCopies(TorchFunctionMode):
    def __init__(self, owner, batch, work):
        self.owner = owner
        self.table = batch.req_to_token_pool.req_to_token
        self.slots = [req.req_pool_idx for req in batch.reqs]
        self.work = work
        self.target = None
        self.writes = []

    def __torch_function__(self, operation, types, args=(), kwargs=None):
        keywords = {} if kwargs is None else kwargs
        if operation is not torch.Tensor.__setitem__:
            return operation(*args, **keywords)
        target, index, source = args
        if (
            not isinstance(source, torch.Tensor)
            or source.untyped_storage()._cdata != self.table.untyped_storage()._cdata
        ):
            return operation(*args, **keywords)

        with self.owner.tensors.metadata():
            if (not isinstance(index, slice) or target.ndim != 1 or target.stride(0) != 1
                    or source.ndim != 1 or source.stride(0) != 1):
                raise ValueError("receiver copy requires contiguous source and destination slices")
            slot, begin = divmod(source.storage_offset() - self.table.storage_offset(), self.table.stride(0))
            start, end, step = index.indices(target.numel())
            count = source.numel()
            if step != 1 or end - start != count or begin + count > self.table.size(1):
                raise ValueError("receiver copy exceeds its row or destination")
            view = target[index]

        position = len(self.writes)
        if position >= len(self.slots) or self.slots[position] != slot or (position and target is not self.target):
            raise ValueError("receiver copy does not follow the admitted rows into one destination")
        member = self.work["members"][position]
        work = {key: self.work[key] for key in ("phase", "rank")}
        work["members"] = [member]

        # Bracket only the native assignment. Reading the resulting KV and synchronizing it belong to qualification.
        with self.owner.capture.call("row_copy", "request_to_token.row_copy", "torch.Tensor.__setitem__",
                                     parameters={"elements": count}, arguments={"elements": "new_tokens"},
                                     work=work, device=True) as call:
            call.record["row_call"] = {"member": position, "parent_port": "cache_locations",
                                       "source_port": "source", "destination_port": "destination"}
            call.port("source", "input", self.owner.tensors.observe(source, ["elements"]), use="state")
            result = operation(*args, **keywords)
            call.port("destination", "output", self.owner.tensors.observe(view, ["elements"]), use="device")

        self.target = target
        self.writes.append({"slot": slot, "source": [begin, begin + count], "target": [start, end]})
        return result

    def readback(self, output):
        if len(self.writes) != len(self.slots) or (self.writes and self.target is not output):
            raise ValueError("receiver did not publish the destination filled by its admitted rows")
        return {"slots": self.slots, "writes": self.writes}


class ServingCuts(CallSegments):
    """Use the native scheduler's same cut points for structure and CPU timing."""

    def __init__(self, aliases, mode, *, effects=False) -> None:
        super().__init__()
        self.aliases = aliases
        self.mode = mode
        self.effects = effects
        self.resources = ResourceCalls(self)
        self.selection = BatchSelection(self)

    def request_id(self, native_id):
        return self.aliases[native_id]

    def batch_work(self, batch):
        return self.work(batch, BatchTensors.members(batch), phase=BatchTensors.phase(batch))

    def completed(self, requests):
        return [
            {"id": self.request_id(req.rid), "generated_tokens": len(req.output_ids), "finished": req.finished()}
            for req in requests
        ]

    def result(self, original, scheduler, batch, output):
        if not self.enabled:
            return original(scheduler, batch, output)
        work = self.batch_work(batch)
        parameters = BatchTensors.parameters(batch, sum(member["new_tokens"] for member in work["members"]))
        with self.call(
            "result", original, parameters=parameters, work=work, host_inputs=HostReads.result_inputs(output)
        ) as frame:
            result = original(scheduler, batch, output)

            # A nested prebuilt result already publishes its own completion boundary.
            if self.effects and not batch.forward_mode.is_prebuilt():
                frame.outputs = self.completed(batch.reqs)
            return result

    def publication(self, original, sender, output, *args, **kwargs):
        BatchTokenIDOutput = resolve(symbols().batch_token_id_output)

        if self.current.get() is None or not isinstance(output, BatchTokenIDOutput) or sender.socket is None:
            return original(sender, output, *args, **kwargs)
        with self.call("publication", original) as frame:
            if self.effects:
                frame.outputs = [
                    {"id": self.request_id(request), "generated_tokens": tokens}
                    for request, tokens in zip(output.rids, output.completion_tokens, strict=True)
                ]
            return original(sender, output, *args, **kwargs)

    def bind(self, result):
        allocator = result.token_to_kv_pool_allocator
        allocator_type = type(allocator)
        allocator._prologue_allocation = AllocationABI(f"{allocator_type.__module__}.{allocator_type.__qualname__}")

        self.resources.bind(result.tree_cache, allocator, result.req_to_token_pool)

    def build(self, original, *args, **kwargs):
        result = original(*args, **kwargs)

        # Diagnostics retain the original readers before attaching the same cuts used on an initialized worker.
        if self.effects:
            self.observe_pools(result, kwargs["server_args"])
        self.bind(result)
        return result

    def install(self) -> None:
        HookTable.around(symbols().build_kv_cache, self.build)
        self.install_calls()

    def install_calls(self) -> None:
        HookTable.group({
            symbols().scheduler_process_batch_result: self.result,
            symbols().sender_wrapper_send_output: self.publication,
        }).install()
        self.selection.install()
        self.resources.install()
        DecisionClocks(self).install()
        HostReads(self).install()
        MetadataCollectives(self).install()
        if self.mode != "null":
            DisaggregatedCalls(self).install(self.mode)


class FlowScopes(ServingCuts):
    def __init__(self, observation) -> None:
        super().__init__(observation.aliases, observation.mode, effects=True)
        self.observation = observation
        self.request_state = observation.request_state
        self.queries = []

    @property
    def enabled(self):
        return self.observation.active and self.observation.selected

    @property
    def protocol_enabled(self):
        return self.observation.protocol_enabled

    @property
    def scheduler(self):
        return self.observation.scheduler

    @property
    def groups(self):
        return self.observation.groups

    @property
    def rank(self):
        return self.observation.parallel["rank"]

    @contextmanager
    def read_boundary(self, kind, original, work, condition):
        owner = self.observation
        with owner.call("query", original, parameters={}, work=work) as call:
            call.record["bindings"].update(action=True, condition=condition)
            query = {"kind": kind, "scope": call.identity}
            if "read" in condition:
                query["read"] = condition["read"]
            with owner.capture.runtime.service_scope(query):
                yield query
            self.queries.append(query)

    def collective(self, original, boundary, arguments):
        owner = self.observation
        collective = boundary.condition["collective"]
        role = collective["group"]
        descriptor = owner.worker_roles[str(owner.parallel["rank"])]["groups"][role]
        members = descriptor["members"]
        root = collective["root"]
        values, outputs = (
            MetadataCollectives.snapshot(arguments, root) if owner.resource_history is not None else (None, None)
        )
        with owner.call(boundary.operation, original, parameters=boundary.parameters, work=boundary.work) as call:
            call.record["bindings"].update(action=True, condition=boundary.condition, collective_group=role)
            call.record["communication"] = {
                "channel": descriptor["channel"], "operation": collective["operation"], "members": members,
                "required": members if root is None else [members[root]], "rank": owner.parallel["rank"], "order": 0,
            }
            result = original(**arguments)

        # Values are optional evidence outside measured work; they never become template parameters.
        if values is not None:
            values["output"] = outputs.flatten().tolist()
            owner.action_values.append({"scope": call.identity, "values": {"collective": values}})
        return result

    @contextmanager
    def boundary(self, original, boundary):
        owner = self.observation
        with owner.call(boundary.operation, original, parameters=boundary.parameters, work=boundary.work,
                        device=boundary.input_use == "device", implementation=boundary.implementation) as call:

            # Snapshot mutation epochs before native work; an early return may later discard unused ports.
            inputs = {name: owner.tensors.observe(tensor, shape, materializer=boundary.materializer)
                      for name, (tensor, shape) in boundary.inputs.items()}
            yield
            call.record["parameters"].update(boundary.parameters)
            call.record["work"]["context"] = boundary.work["context"]
            call.record["bindings"].update(action=True, condition=boundary.condition)
            call.record["guards"] = list(boundary.guards)
            for name in boundary.inputs:
                call.port(name, "input", inputs[name], use=boundary.input_use, at_return=boundary.input_use == "host")
            for name, (tensor, shape) in boundary.outputs.items():
                call.port(name, "output", owner.tensors.observe(tensor, shape), use="device")

    @contextmanager
    def enclosing(self, frame):
        # Segments and child calls own every device operation; this envelope preserves only their nesting.
        with self.observation.call(
            "invocation", frame.original, parameters=frame.parameters, work=frame.work,
        ) as call:
            call.record["envelope"] = True
            if frame.operation == "prebuilt_prepare":
                table = ReceiverTableCopies(self.observation, frame.inputs, frame.work)
                with table:
                    yield

                # The last continuation publishes the completed table; the observation spans the entire native loop.
                terminal = self.observation.capture.scopes[frame.observations[-1]]
                terminal["page_table"] = table.readback(frame.inputs.out_cache_loc)
            else:
                yield
            if frame.local_upload is not None:
                call.record["parameters"][frame.local_upload] = frame.parameters[frame.local_upload]
        if frame.values:
            self.observation.action_values.append({
                "scope": frame.observations[-1], "values": {**frame.values, "segments": frame.observations},
            })

    @contextmanager
    def fragment(self, frame):
        owner = self.observation
        with owner.call(
            "segment", frame.original, device=not frame.host_only and frame.operation != "kv_enqueue"
        ) as call:
            call.record["guards"] = list(frame.guards)
            for name, (tensor, shape) in frame.tensor_inputs.items():
                call.port(name, "input", owner.tensors.observe(tensor, shape), use="device")
            for name, (tensor, shape) in frame.metadata_inputs.items():
                call.port(name, "input", owner.tensors.observe(tensor, shape), use="metadata")
            copies = {name: owner.tensors.local_copy(*operand) for name, operand in frame.local_copies.items()}
            upload = TensorUpload(owner.tensors, frame.local_upload) if frame.local_upload is not None else None
            with upload if upload is not None else nullcontext():
                yield
            if upload is not None:
                if upload.operand is None:
                    raise ValueError("native KV release did not upload its deduplicated page indices")

                # The page count is only known after native deduplication. Bind it where the copy runs.
                parameter = frame.local_upload
                extent, = upload.operand["shape"]
                frame.parameters[parameter] = extent
                call.record["parameters"][parameter] = extent
                call.record["arguments"][parameter] = parameter
                copies[parameter] = upload.operand
            if copies:
                call.record["local_copies"] = copies
            call.record["work"]["continuation"] = frame.continuation
            call.record["bindings"]["segment"] = {
                "owner": frame.operation,
            }
            call.record["bindings"]["condition"] = frame.observed_condition
            if frame.host_fragment:
                call.record["bindings"]["segment"]["host_only"] = True
            for name, (tensor, shape) in frame.tensor_outputs.items():
                call.output(name, owner.tensors.observe(tensor, shape), use="device", preceding=frame.observations)
            if frame.terminal and frame.operation in {"result", "prebuilt_result", "prefill_finish"}:
                call.record["effects"] = frame.outputs
            if frame.terminal and frame.operation == "prebuilt_prepare" and owner.resource_history is not None:
                owner.resource_history.batch(call.record["id"], frame.inputs)
            if frame.terminal and frame.operation == "publication":
                call.record["publication"] = frame.outputs
                call.port("publication", "output", None, use="host")

        # Segment identities also connect deferred tensor outputs when optional diagnostic recording is disabled.
        frame.observations.append(call.identity)

    def observe_pools(self, result, server_args):
        allocator = result.token_to_kv_pool_allocator

        # A PD receiver can finish without a model call; pool geometry belongs to construction, not attention.
        self.observation.worker.pools(result, server_args)
        self.observation.kv_layout = self.observation.worker.kv_layout
        self.observation.prepare_roles()
        if self.observation.resource_history is not None:
            self.observation.resource_history.bind(result.tree_cache, allocator, result.req_to_token_pool)


class ObservedExecutor:
    def __init__(self, executor) -> None:
        self.executor = executor

    def submit(self, operation, *args, **kwargs):
        # Native workers remain native; only the observed caller identity follows each submitted job.
        return self.executor.submit(copy_context().run, operation, *args, **kwargs)


class ReservationSocket:
    def __init__(self, owner, socket, receiver, peer) -> None:
        self.owner, self.socket, self.receiver, self.peer = owner, socket, receiver, peer

    def send_multipart(self, *args, **kwargs):
        return self.owner.send_reservation(self.socket.send_multipart, self.receiver, self.peer, args, kwargs)


class NativeKvScopes:
    def __init__(self, calls, observation=None) -> None:
        self.calls = calls
        self.observation = observation
        self.chunk = ContextVar("native_kv_chunk", default=None)
        self.service = ContextVar("native_kv_service", default=None)
        self.request = ContextVar("native_kv_request", default=None)
        self.reservation = ContextVar("native_kv_reservation", default=None)
        self.manager = None
        self.reservations = {}
        self.receipts = []
        self.states = []

    def update_status(self, original, manager, bootstrap_room, status):
        owner = self.observation
        request = owner.description["protocol_requests"].get(str(bootstrap_room))
        if not owner.selected or request is None:
            return original(manager, bootstrap_room, status)

        # Source rooms are known before profiler activation; warmup rooms are absent from this submission map.
        record = {"request": request}
        with owner.capture.runtime.service_scope(record):
            record["before"] = manager.request_status.get(bootstrap_room)
            result = original(manager, bootstrap_room, status)
            record["after"] = manager.request_status.get(bootstrap_room)
        self.states.append(record)
        return result

    def connect_reservation(self, original, cls, peer):
        socket, lock = original(cls, peer)
        receiver = self.reservation.get()
        if receiver is not None:
            socket = ReservationSocket(self, socket, receiver, peer)
        return socket, lock

    def receive_reservation(self, original, cls, message):
        result = original(cls, message)
        owner = self.observation
        request = owner.description["protocol_requests"].get(str(result.room))
        if owner.selected and request is not None:

            # The receiver inserts the parsed contribution before reading the next reservation.
            self.receipts.append({"request": request, "session": result.mooncake_session_id,
                                  "received_raw_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW)})
        return result

    def send_reservation(self, original, receiver, peer, args, kwargs):
        work = self.calls.request_work(receiver._prologue_request)
        work["context"]["reservation_peer"] = receiver.bootstrap_infos.index(peer)
        owner = self.observation
        with self.calls.separate("kv_reservation", work):
            if owner is None:
                with self.calls.native_call("kv_reservation", original, work=work):
                    return original(*args, **kwargs)

            with owner.call("kv_reservation", original, parameters={}, work=work) as call:
                scope = call.record
                scope["bindings"]["action"] = True

                # Remote visibility can begin as soon as the complete payload is handed to the socket.
                scope["ports"].append({"name": "reservation.ready", "direction": "output", "use": "signal",
                                       "boundary": {"scope": scope["id"], "interval": "call", "side": "begin"}})
                publication = {"request": work["context"]["request"]["id"],
                    "target": [peer["rank_ip"], peer["rank_port"]]}
                scope["reservation_publication"] = publication
                with owner.capture.runtime.service_scope(publication):
                    return original(*args, **kwargs)

    def poll(self, original, transport):
        if not self.calls.enabled:
            return original(transport)

        work = self.calls.request_work(transport._prologue_request)
        with self.calls.read("transport", original, work) as query:
            result = original(transport)
            query["result"] = result
        return result

    def reserve(self, original, receiver, kv_indices, aux_index=None, state_indices=None, decode_prefix_len=None):
        owner = self.observation
        request = receiver._prologue_request
        if owner is not None:
            if owner.resource_history is not None:
                owner.resource_history.state_binding("metadata", request, aux_index)
            if request.rid in self.calls.aliases:
                self.reservations[str(receiver.bootstrap_room)] = {"request": self.calls.request_id(request.rid),
                    "index": aux_index}
        if not self.calls.enabled:
            return original(receiver, kv_indices, aux_index, state_indices, decode_prefix_len)

        work = {"context": {"request": self.calls.progress((request,))[0]}}
        with self.calls.call("kv_reserve", original, parameters={"transfer_pages": len(kv_indices)}, work=work):
            previous = self.reservation.set(receiver)
            try:
                return original(receiver, kv_indices, aux_index, state_indices, decode_prefix_len)
            finally:
                self.reservation.reset(previous)

    def metadata(self, original, buffers, req):
        aliases = self.observation.content.aliases
        if req.bootstrap_room is not None and req.rid in aliases:
            self.reservations[str(req.bootstrap_room)] = {"request": aliases[req.rid],
                "index": req.metadata_buffer_index}
        if self.observation.resource_history is not None:
            self.observation.resource_history.state_binding("metadata", req, req.metadata_buffer_index)
        return original(buffers, req)

    def metadata_allocator(self, original, allocator, size):
        original(allocator, size)
        owner = self.observation
        if owner is not None and owner.resource_history is not None:
            owner.resource_history.bind_field("metadata", allocator)
        self.calls.resources.bind_field("metadata", allocator)

    def manager_init(self, original, manager, *args, **kwargs):
        result = original(manager, *args, **kwargs)
        self.manager = manager
        return result

    def send_chunk(self, original, scheduler, req, *args, **kwargs):
        previous = self.request.set((req, scheduler.token_to_kv_pool_allocator.page_size))
        try:
            return original(scheduler, req, *args, **kwargs)
        finally:
            self.request.reset(previous)

    def enqueue(self, original, queue, item):
        TransferKVChunk = resolve(symbols().transfer_kv_chunk)

        owner = self.observation
        if not self.calls.enabled or not isinstance(item, TransferKVChunk):
            return original(queue, item)
        request, page_size = self.request.get()
        begin = request.start_send_idx // page_size * page_size
        end = min(begin + len(item.prefill_kv_indices) * page_size, len(request.origin_input_ids))
        work = self.calls.request_work(request, begin, end)
        parameters = {"transfer_pages": len(item.prefill_kv_indices), "transfer_tokens": end - begin,
            "last_chunk": int(item.is_last_chunk)}

        # FastQueue uses a reentrant condition: publish the completed marker before a consumer can take the task.
        with queue._cond:
            with self.calls.call(
                "kv_enqueue", original, parameters=parameters, work=work,
                guards=(
                    ["eq", "last_chunk", parameters["last_chunk"]],
                    ["lt", 0, "transfer_pages"] if len(item.prefill_kv_indices) else ["eq", "transfer_pages", 0],
                ),
            ):
                if owner is not None:
                    call = owner.capture.scopes[owner.capture.current.get()]
                    item._prologue_submission = call["id"]
                    call["native_kv_chunk"] = {
                        "room": item.room, "source_pages": item.prefill_kv_indices.tolist(),
                        "metadata_index": item.prefill_aux_index, "request_slot": int(request.req_pool_idx) - 1,
                    }
                return original(queue, item)

    def dequeue(self, original, queue):
        self.chunk.set(None)
        item = original(queue)
        if hasattr(item, "_prologue_submission"):
            self.chunk.set(item)
        return item

    def transfer(self, signature, original, manager, *args, **kwargs):
        owner = self.observation
        chunk = self.chunk.get()
        if not owner.active or not owner.selected or chunk is None:
            return original(manager, *args, **kwargs)
        arguments = signature.bind(manager, *args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        if "executor" in values:
            values["executor"] = ObservedExecutor(values["executor"])

        with owner.capture.call(
            "kv_transfer", original.__qualname__, f"{original.__module__}.{original.__qualname__}",
            parent=chunk._prologue_submission, service={"kind": "transfer", "role": "kv_transport", "executions": []},
        ) as call:
            call.record["submitted_by"] = chunk._prologue_submission
            call.record["native_transfer"] = {
                "kind": "metadata" if original.__name__ == "send_aux" else "kv", "blocks": [],
                "session": values["req"].mooncake_session_id
                    if original.__name__ == "send_aux" else values["mooncake_session_id"],
            }
            if original.__name__ != "send_aux":
                call.record["native_transfer"].update(
                    source_pages=values["prefill_kv_indices"].tolist(), target_pages=values["dst_kv_indices"].tolist(),
                )
            else:
                call.record["native_transfer"].update(
                    room=values["req"].room, source_index=values["prefill_aux_index"],
                        target_index=values["req"].dst_aux_index,
                )

                # The submitted request owns both metadata endpoints; interpretation only resolves their physical rows.
                member, = call.record["work"]["members"]
                call.record["work"].update({f"metadata/{mode}": [member["id"]] for mode in ("read", "write")})
            previous = self.service.set(call.record)
            try:
                result = original(*arguments.args, **arguments.kwargs)
                if result != 0:
                    raise RuntimeError(f"native KV service failed: {original.__qualname__}: {result}")
            finally:
                self.service.reset(previous)

        # Publish observed geometry after the measured service returns so processing the evidence adds no cost.
        self.payload(call.record)
        return result

    @staticmethod
    def payload(scope):
        transfer = scope["native_transfer"]
        transferred_bytes = sum(length for _, _, length in transfer["blocks"])
        scope["service"]["payload"] = {"bytes": transferred_bytes, "direction": "peer_device"}
        scope["parameters"] = {"transfer_bytes": transferred_bytes}

        # State is indexed by logical tokens; the physical transport sends full pages or fixed metadata rows.
        pages = len(transfer.get("source_pages", ()))
        if pages:
            page_bytes, tail = divmod(transferred_bytes, pages)
            if tail or not page_bytes:
                raise ValueError("native KV transport has no complete per-page byte extent")
            scope["arguments"] = {"transfer_bytes": ["mul", "transfer_pages", page_bytes]}
            scope["constants"] = {}
        else:
            scope["arguments"] = {}
            scope["constants"] = {"transfer_bytes": transferred_bytes}

    def transfer_data(self, original, engine, session_id, buffers, peer_buffer_addresses, lengths):
        call = self.service.get()
        if call is None:
            return original(engine, session_id, buffers, peer_buffer_addresses, lengths)
        execution = {}
        with self.observation.capture.lock:
            call["native_transfer"]["blocks"].extend([
                [int(source), int(target), int(length)]
                for source, target, length in zip(buffers, peer_buffer_addresses, lengths, strict=True)
            ])
            call["service"]["executions"].append(execution)
        with self.observation.capture.runtime.service_execution(call["id"], execution):
            return original(engine, session_id, buffers, peer_buffer_addresses, lengths)

    def registration(self):
        if self.manager is None:
            return None
        args = self.manager.kv_args
        return {
            "session": self.manager.engine.session_id,
            "control_endpoint": [self.manager.local_ip, self.manager.rank_port],
            "pointers": [int(value) for value in args.kv_data_ptrs],
            "lengths": [int(value) for value in args.kv_data_lens],
            "page_bytes": [int(value) for value in args.kv_item_lens],
            "metadata": {
                "pointers": [int(value) for value in args.aux_data_ptrs],
                "lengths": [int(value) for value in args.aux_data_lens],
                "row_bytes": [int(value) for value in args.aux_item_lens],
                "reservations": self.reservations,
            },
        }

    def install(self):
        HookTable.group({
            symbols().req_to_metadata_idx_allocator_init: self.metadata_allocator,
            symbols().connect_to_bootstrap_server: self.connect_reservation,
            symbols().mooncake_kv_sender_poll: self.poll,
            symbols().mooncake_kv_receiver_poll: self.poll,
            symbols().scheduler_send_kv_chunk: self.send_chunk,
            symbols().mooncake_kv_receiver_send_metadata: self.reserve,
            symbols().fast_queue_put: self.enqueue,
        }).install()
        if self.observation is None:
            return

        DeviceKVManager = resolve(symbols().device_kv_manager)
        MooncakeKVManager = resolve(symbols().mooncake_kv_manager)

        HookTable.group({
            symbols().common_kv_manager_update_status: self.update_status,
            symbols().mooncake_kv_manager_init: self.manager_init,
            symbols().transfer_info_from_zmq: self.receive_reservation,
            symbols().metadata_buffers_set_buf: self.metadata,
            symbols().fast_queue_get: self.dequeue,
            symbols().mooncake_transfer_engine_batch_transfer_sync: self.transfer_data,
        }).install()
        for cls, method in (
            (DeviceKVManager, "send_kvcache"),
            (MooncakeKVManager, "send_kvcache_slice"),
            (MooncakeKVManager, "send_aux"),
        ):
            HookTable.around(
                f"{cls.__module__}.{cls.__qualname__}.{method}",
                partial(self.transfer, inspect.signature(getattr(cls, method))),
            )
