"""Drive native PD admission and transfer using the core resource and delivery protocol."""

from collections import defaultdict
from functools import partial
from threading import Lock

import torch

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.control.resources import StateSlots
from prologue.integrations.frameworks.sglang.shared.batches import RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import PdPeerBindings


class PdTopology:
    def __init__(self, workers, local):
        instances = {source["id"]: source for source, _, _, _ in workers}
        prefill, = [source for source in instances.values() if source["mode"] == "prefill"]
        decode, = [source for source in instances.values() if source["mode"] == "decode"]
        if (
            len(instances) != 2
            or prefill["model_config"] != decode["model_config"]
            or prefill["dtype"] != decode["dtype"]
        ):
            raise ValueError("PD requires one Prefill and one Decode instance of the same model and dtype")
        rows = {
            index: {
                "mode": source["mode"],
                "parallel": {
                    "logical_tp": source["tp"], "logical_pp": source["pp"],
                    "tp_rank": rank % source["tp"], "pp_rank": rank // source["tp"],
                },
                "layout": source["workers"][rank]["kv_layout"], "page_size": settings["page_size"],
            }
            for index, (source, settings, _, rank) in enumerate(workers)
        }
        members = [source["workers"][rank] for source, _, _, rank in workers]
        bindings = PdPeerBindings(rows)
        self.peers = {}
        self.info = bindings.infos.get(local)
        for origin, targets in bindings.destinations.items():
            layouts = bindings.layouts(origin)
            for target, binding in targets.items():
                if local not in (origin, target):
                    continue
                if members[origin]["kv_peers"][binding] != layouts[binding]:
                    raise ValueError("PD destination geometry differs from the observed transfer template")
                peer = target if local == origin else origin
                self.peers[peer] = {
                    "pool": members[target]["group"], "binding": binding, "group": members[peer]["group"],
                }
                if local == target:
                    self.peers[peer]["publishers"] = [members[index]["group"] for index in targets]

        if local in bindings.sources:

            # Use the native bootstrap order: TP order, then higher PP stages first.
            self.peers = {peer: self.peers[peer] for peer in bindings.sources[local]}


class MetadataRows:
    def __init__(self, buffers):
        pointers, _, _ = buffers.get_buf_infos()
        tensors = {value.data_ptr(): value for value in vars(buffers).values() if isinstance(value, torch.Tensor)}
        self.buffers = [tensors[pointer] for pointer in pointers]

    def read(self, index):
        # Preserve native values, including its room gate; copying them here does not make them ready in simulated time.
        return [buffer[index].tolist() for buffer in self.buffers]

    def write(self, index, values):
        for buffer, row in zip(self.buffers, values, strict=True):
            if len(row) != buffer.shape[1]:
                raise ValueError("PD metadata row layout differs between native participants")
            buffer[index].copy_(buffer.new_tensor(row))


BaseKVManager = resolve(symbols().base_kv_manager)
KVArgs = resolve(symbols().kv_args)
KVPoll = resolve(symbols().kv_poll)
CommonKVManager = resolve(symbols().common_kv_manager)
CommonKVReceiver = resolve(symbols().common_kv_receiver)
MooncakeKVReceiver = resolve(symbols().mooncake_kv_receiver)
MooncakeKVSender = resolve(symbols().mooncake_kv_sender)
KVClassType = resolve(symbols().kv_class_type)
envs = resolve(symbols().envs)


class NativePdBackend:
    def __init__(self, scheduler, workers, local):
        self.scheduler = scheduler
        self.topology = PdTopology(workers, local)
        self.local = local
        self.senders = {}
        self.receivers = {}
        self.reservations = defaultdict(dict)
        self.published = {}
        self.sequence = 0
        self.failures = {}

    def exchange(self, deliveries=(), *, completion=None):
        for packet in self.scheduler.peer.exchange(
            "messages", {"deliveries": list(deliveries), "completion": completion}
        ):
            source, value = packet["source"], packet["value"]
            if source not in self.topology.peers and source != self.local:
                raise ValueError("PD message came from a worker outside its native transfer topology")
            kind = value["kind"]
            if kind == "reservation":
                self.reservations[value["room"]][source] = value
            elif kind == "transfer":
                self.receivers[value["room"]].pending[source] = value
            elif kind in ("reservation_visible", "sender_visible", "receiver_visible"):
                room = value["room"]
                state = KVPoll.WaitingForInput if kind == "reservation_visible" else KVPoll.Success

                # A reservation can become visible before the local scheduler creates its sender.
                self.manager.status_events[room][state] = packet["completion"]
                if kind != "receiver_visible":
                    self.manager.update_status(room, state)
            elif kind == "consumed":
                publication = self.published[value["publication"]]
                publication["pending"].remove(source)
                if not publication["pending"]:
                    del self.published[value["publication"]]
            elif kind == "failed":
                self.failures[value["room"]] = value["reason"]
            else:
                raise ValueError(f"unknown PD control message: {kind}")

    def publish(self, deliveries, req, notice, *, signals=(), after=()):
        self.exchange(deliveries, completion={
            "request": RequestSubmissions.index(req.rid), "notification": notice,
            "signals": list(signals), "after": list(after),
        })

    @staticmethod
    def notice(group, name, request, **work):
        return {"group": group, "name": name, "work": {"request": request, **work}}

    def live_signals(self):
        signals = {signal for record in self.published.values() for signal in record["signals"]}
        for transport in (*self.senders.values(), *self.receivers.values()):
            signals.update(transport.signals)
        return signals

    def waiting(self):
        transports = (*self.senders.values(), *self.receivers.values())

        # Local readiness may still await PP consensus; only native queues can turn it into runnable work.
        return bool(transports) and all(
            transport.poll_state in (KVPoll.Bootstrapping, KVPoll.WaitingForInput, KVPoll.Transferring, KVPoll.Success)
            for transport in transports
        )

    def failed(self, room):
        self.exchange()
        return room in self.failures

    def abort(self, room):
        reason = self.failures.setdefault(room, f"native PD request {room} was aborted")
        self.exchange([
            {"target": peer, "value": {"kind": "failed", "room": room, "reason": reason}}
            for peer in self.topology.peers
        ])

    @staticmethod
    def group_poll(original, pollers, *args, **kwargs):
        results = original(pollers, *args, **kwargs)

        # Keep the decision already consumed by native queues, including metadata gates and TP/CP agreement.
        for poller, result in zip(pollers, results, strict=True):
            poller.poll_state = result
        return results

    def install_resources(self):
        decode = resolve(symbols().decode)
        prefill = resolve(symbols().prefill)
        scheduler = resolve(symbols().scheduler_module)

        # Replace the transport factory in this worker only; native queues and their control methods remain intact.
        decode.get_kv_class = prefill.get_kv_class = self.native_class
        scheduler.ReqToMetadataIdxAllocator = partial(StateSlots, peer=self.scheduler.peer, field="metadata")

    def install_calls(self):
        self.scheduler.calls.resources.bind_field("metadata", self.scheduler.req_to_metadata_buffer_idx_allocator)

        # Serving hooks must join the graph wrappers before the registry patches their shared targets.
        HookTable.around(symbols().metadata_buffers_set_buf, self.bind_metadata)
        for name in ("poll_and_all_reduce", "poll_and_all_reduce_attn_cp_tp_group"):
            HookTable.around(f"{symbols().utils}.{name}", self.group_poll)

    def bind_metadata(self, original, buffers, req):
        # Native allocation precedes request admission; bind the row before its first graph write.
        self.scheduler.req_to_metadata_buffer_idx_allocator.bind(req, req.metadata_buffer_index)
        return original(buffers, req)

    def native_class(self, backend, kind):
        return {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: partial(NativeKvManager, backend=self),
            KVClassType.SENDER: NativeKvSender,
            KVClassType.RECEIVER: NativeKvReceiver,
        }[kind]


class NativeKvManager(BaseKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False, *, backend):
        if is_mla_backend or args.state_types:
            raise ValueError("PD templates require the observed MHA pool without auxiliary state pools")
        self.backend = backend
        backend.manager = self
        self.kv_args = args
        self.server_args = server_args
        self.is_dummy_cp_rank = False
        self.enable_all_cp_ranks_for_transfer = False
        self.request_status = {}
        self.status_events = defaultdict(dict)
        self.addr_to_rooms_tracker = defaultdict(set)
        self.failure_records = backend.failures
        self.failure_lock = Lock()
        self.enable_trace = False
        self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        self.req_to_decode_prefix_len = {}
        self.prefill_info_table = {}
        self.kv_item_lens_sum = sum(args.kv_item_lens)
        self.state_item_lens_sum = 0
        self.rows = MetadataRows(backend.scheduler.disagg_metadata_buffers)

    check_status = CommonKVManager.check_status
    update_status = CommonKVManager.update_status

    def register_to_bootstrap(self):
        # The serving task already supplies both native worlds; device-free control does not open a bootstrap server.
        return None

    def try_ensure_parallel_info(self, address):
        self.prefill_info_table[address] = self.backend.topology.info
        return True

    def record_failure(self, room, reason):
        self.backend.failures[room] = reason


class NativeKvPolling:
    def packet(self, target, kind, **fields):
        return {"target": target, "value": {"kind": kind, "room": self.bootstrap_room, **fields}}

    def poll(self):
        calls = self.backend.scheduler.calls
        if not calls.enabled:
            self.poll_state = self._poll()
            return self.poll_state

        with calls.read("transport", self.poll, calls.request_work(self._prologue_request)) as query:
            self.poll_state = self._poll()

            # Record only the state actually read, not unrelated deliveries collected in the same inbox drain.
            completions = self.kv_mgr.status_events.get(self.bootstrap_room, {})
            visible = [state for state in completions if state <= self.poll_state]
            if visible:
                self.backend.scheduler.peer.call(
                    "observed_completion", event=completions[max(visible)], boundary=query["boundary"],
                )
            query["result"] = self.poll_state
        return self.poll_state


class NativeKvSender(NativeKvPolling, MooncakeKVSender):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.backend = mgr.backend
        self.signals = set()
        self.poll_state = None
        self.backend.senders[bootstrap_room] = self

    def _poll(self):
        if self.backend.failed(self.bootstrap_room):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        state = self.kv_mgr.check_status(self.bootstrap_room)
        if self.conclude_state is None and state == KVPoll.WaitingForInput and self.aux_index is None:
            reservations = self.backend.reservations[self.bootstrap_room]
            prefixes = {value["prefix"] for value in reservations.values()}
            if len(prefixes) != 1:
                raise ValueError("native PD destinations disagree on the reused prefix")
            self.kv_mgr.req_to_decode_prefix_len[self.bootstrap_room] = prefixes.pop()

        # Native poll owns terminal states and bootstrap timeout; only transport visibility is adapted.
        return MooncakeKVSender.poll(self)

    def send(self, kv_indices, state_indices=None):
        if state_indices:
            raise ValueError("PD transfer template has no auxiliary state-pool contract")
        kv_indices, index_slice, last, _ = self._prepare_send_indices(kv_indices, state_indices)
        self._record_transfer_indices(kv_indices, state_indices)
        backend, scheduler = self.backend, self.backend.scheduler
        req = scheduler.calls.current.get().inputs
        page_size = self.kv_mgr.kv_args.page_size
        begin = req.start_send_idx // page_size * page_size
        end = min(begin + len(kv_indices) * page_size, len(req.origin_input_ids))
        identity = backend.sequence
        backend.sequence += 1
        signals, packets, pools = {}, [], {}
        if len(kv_indices) or last:
            signals["transfer.complete"] = f"{scheduler.group}/pd/{identity}"
        metadata = self.kv_mgr.rows.read(self.aux_index) if last else None
        reservations = backend.reservations[self.bootstrap_room]
        for target, destination in reservations.items():
            peer = backend.topology.peers[target]
            if len(destination["pages"][index_slice]) != len(kv_indices):
                raise ValueError("native PD sender chunk exceeds its receiver's page reservation")
            pools[peer["binding"]] = peer["pool"]
            if last:
                packets.append(self.packet(target, "transfer", publication=identity, metadata=metadata))
        self.signals.update(signals.values())

        # Native Success is published after the whole destination group finishes, not each destination independently.
        if last:
            completion_signals = sorted(self.signals)
            packets.append(self.packet(backend.local, "sender_visible"))

            # The native transport only notifies after the final chunk; retain all chunks until that notice is consumed.
            backend.published[identity] = {"signals": completion_signals, "pending": set(backend.topology.peers)}
        work = {
            **scheduler.calls.request_work(req, begin, end),
            "pools": pools, "signals": signals,
            "bindings": {"transport_order": [backend.topology.peers[target]["pool"] for target in reservations]},
        }
        parameters = {"transfer_pages": len(kv_indices), "transfer_tokens": end - begin, "last_chunk": int(last)}

        # Register before the call returns: background publication can finish while its enclosing CPU work is still
        # running.
        if last:
            request = scheduler.calls.progress((req,))[0]["id"]
            backend.publish(packets, req, backend.notice(scheduler.group, "kv_success", request),
                            signals=completion_signals)
        with scheduler.calls.call("kv_enqueue", self.send, parameters=parameters, work=work):
            if last:
                scheduler.request_state.note("metadata", "read", [req])
                scheduler.request_state.note("metadata", "write", [req])

    def clear(self):
        super().clear()
        self.kv_mgr.status_events.pop(self.bootstrap_room, None)
        del self.backend.senders[self.bootstrap_room]
        self.backend.reservations.pop(self.bootstrap_room, None)

    def abort(self):
        super().abort()
        self.backend.abort(self.bootstrap_room)


class NativeKvReceiver(NativeKvPolling, MooncakeKVReceiver):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room=None):
        CommonKVReceiver.__init__(self, mgr, bootstrap_addr, bootstrap_room)
        self.backend = mgr.backend
        self.poll_state = None
        self.pending = {}
        self.finished = set()
        self.metadata = None
        self.aux_index = None
        self.signals = set()
        self.backend.receivers[bootstrap_room] = self
        mgr.try_ensure_parallel_info(bootstrap_addr)

    def init(self, prefill_dp_rank):
        if prefill_dp_rank != 0:
            raise ValueError("PD control has no data-parallel bootstrap mapping")
        self.bootstrap_infos = tuple(self.backend.topology.peers.values())
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None, decode_prefix_len=None):
        if state_indices:
            raise ValueError("PD transfer template has no auxiliary state-pool contract")
        backend = self.backend
        scheduler, req = backend.scheduler, self._prologue_request
        calls = scheduler.calls
        scheduler.req_to_metadata_buffer_idx_allocator.bind(req, aux_index)
        work = calls.work(None)
        work["context"] = {"request": calls.progress((req,))[0]}
        request = work["context"]["request"]["id"]
        with calls.call("kv_reserve", self.send_metadata, parameters={"transfer_pages": len(kv_indices)}, work=work):
            self.aux_index = aux_index
            pages = kv_indices.tolist()
            for index, (peer, target) in enumerate(backend.topology.peers.items()):
                recipient = target["group"]
                signal = f"{scheduler.group}/pd-reservation/{request}/{recipient}"
                self.signals.add(signal)
                receipt = partial(backend.notice, recipient, "kv_reservation_received", request)
                packet = self.packet(peer, "reservation", pages=pages, prefix=decode_prefix_len or 0)
                backend.publish([packet], req, receipt(publisher=scheduler.group), signals=[signal])

                # Native readiness follows receipt of every reservation in publisher order.
                if scheduler.group == target["publishers"][0]:
                    backend.publish([self.packet(peer, "reservation_visible")], req,
                                    backend.notice(recipient, "kv_ready", request),
                                    after=[receipt(publisher=publisher) for publisher in target["publishers"]])

                publication = calls.request_work(req)
                publication["context"]["reservation_peer"] = index
                publication["signals"] = {"reservation.ready": signal}
                with calls.separate("kv_reservation", publication):
                    scheduler.action("kv_reservation", {}, publication)
            self.init_time = calls.clocks.read("time", symbols().mooncake_kv_receiver_send_metadata)

        # Register before polling: native receive-side publication progresses independently of the next scheduler turn.
        backend.publish([self.packet(backend.local, "receiver_visible")], req,
                        backend.notice(scheduler.group, "kv_success", request),
                        after=[backend.notice(peer["group"], "kv_success", request)
                               for peer in backend.topology.peers.values()])

    def _poll(self):
        if self.backend.failed(self.bootstrap_room):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if (
            self.conclude_state is None
            and self.aux_index is not None
            and self.kv_mgr.check_status(self.bootstrap_room) != KVPoll.Failed
        ):
            acknowledgements = []
            for source, packet in self.pending.items():

                # Delivery crossed core completion; the receiver joins native producers and writes their metadata once.
                if self.metadata is not None and self.metadata != packet["metadata"]:
                    raise ValueError("native PD producers disagree on the first-token metadata")
                self.metadata = packet["metadata"]
                self.finished.add(source)
                acknowledgements.append(
                    {"target": source, "value": {"kind": "consumed", "publication": packet["publication"]}}
                )
            self.pending.clear()
            if acknowledgements:
                self.backend.exchange(acknowledgements)

            visible = KVPoll.Success in self.kv_mgr.status_events.get(self.bootstrap_room, {})
            if visible and self.finished == self.backend.topology.peers.keys():
                self.kv_mgr.rows.write(self.aux_index, self.metadata)
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)

        # Waiting timeout and terminal-state handling remain the installed native policy.
        return MooncakeKVReceiver.poll(self)

    def clear(self):
        del self.backend.receivers[self.bootstrap_room]
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.status_events.pop(self.bootstrap_room, None)
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(self.bootstrap_room)

    def _send_abort_notification(self):
        self.backend.abort(self.bootstrap_room)
