"""Run the native PD transport interfaces against simulated transfer readiness."""

from contextlib import contextmanager
from functools import partial

import torch
from sglang.srt.disaggregation import decode, prefill
from sglang.srt.disaggregation.base import KVArgs, KVPoll
from sglang.srt.disaggregation.base.conn import KVTransferMetric
from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.utils import KVClassType


class Transport:
    def __init__(self, peer, configuration):
        self.peer = peer
        self.prefill = configuration.instances["prefill"]
        self.rooms = {}
        self.next_room = 1
        self.buffers = None

    def room(self, request):
        room = self.next_room
        self.next_room += 1
        self.rooms[room] = request
        return room

    @contextmanager
    def install(self):
        decode_factory, prefill_factory = decode.get_kv_class, prefill.get_kv_class
        decode.get_kv_class = prefill.get_kv_class = self.native_class
        try:
            yield
        finally:
            decode.get_kv_class, prefill.get_kv_class = decode_factory, prefill_factory

    def native_class(self, backend, kind):
        return {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: partial(Manager, transport=self),
            KVClassType.SENDER: Sender,
            KVClassType.RECEIVER: Receiver,
        }[kind]

    def source_row(self, slot):
        return {"token": self.buffers.output_ids[slot, 0].item(), "cached": self.buffers.cached_tokens[slot].tolist(),
                "room": self.buffers.bootstrap_room[slot, 0].item()}

    def write_row(self, slot, metadata):
        self.buffers.output_ids[slot, 0] = metadata["token"]
        self.buffers.cached_tokens[slot] = torch.tensor(metadata["cached"], dtype=self.buffers.cached_tokens.dtype)
        self.buffers.bootstrap_room[slot, 0] = metadata["room"]


class Manager:
    def __init__(self, args, role, server_args, is_mla_backend=False, *, transport):
        if is_mla_backend or args.state_types:
            raise ValueError("native PD control requires the observed dense KV pool")
        self.transport, self.kv_args = transport, args
        self.prefill_info_table = {}

    def try_ensure_parallel_info(self, address):
        source = self.transport.prefill
        self.prefill_info_table[address] = PrefillServerInfo(
            attn_tp_size=source.tp, attn_cp_size=1, dp_size=1, pp_size=source.pp,
            page_size=source.page_size, kv_cache_dtype=None, follow_bootstrap_room=False,
        )
        return True


class Endpoint:
    def __init__(self, mgr, room):
        self.transport, self.room = mgr.transport, room
        self.request = self.transport.rooms[room]

    def call(self, operation, **fields):
        return self.transport.peer.call(operation, request=self.request, **fields)

    def clear(self):
        self.call("retire_handoff")
        del self.transport.rooms[self.room]

    def failure_exception(self):
        raise RuntimeError(f"native PD transfer failed for {self.request}")

    abort = failure_exception

    @staticmethod
    def pages(indices):
        # Native KV metadata skips the dummy page. Core pools are zero-based.
        return [int(page) - 1 for page in indices]


class Sender(Endpoint):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank):
        super().__init__(mgr, bootstrap_room)

    def init(self, num_kv_indices, aux_index):
        self.page_count, self.slot = num_kv_indices, aux_index

    @staticmethod
    def pop_decode_prefix_len():
        return 0

    @staticmethod
    def should_send_kv_chunk(num_pages, last_chunk):
        if not last_chunk:
            raise ValueError("native PD control needs a complete single Prefill offer")
        return True

    @staticmethod
    def get_transfer_metric():
        return KVTransferMetric()

    def poll(self):
        return {
            "bootstrapping": KVPoll.Bootstrapping,
            "reserved": KVPoll.WaitingForInput,
            "transferring": KVPoll.Transferring,
            "ready": KVPoll.Success,
        }[self.call("handoff_status")["state"]]

    def send(self, kv_indices, state_indices=None):
        if state_indices or len(kv_indices) != self.page_count:
            raise ValueError("native PD offer does not match the initialized KV extent")
        self.call("send_handoff", pages=self.pages(kv_indices), metadata_slot=self.slot,
                  metadata=self.transport.source_row(self.slot))


class Receiver(Endpoint):
    require_staging = False

    def __init__(self, mgr, bootstrap_addr, bootstrap_room):
        super().__init__(mgr, bootstrap_room)
        self.slot = None
        self.conclude_state = None
        self.initialized = False

    def init(self, prefill_dp_rank):
        self.initialized = True

    def send_metadata(self, kv_indices, aux_index, state_indices=None, decode_prefix_len=None):
        if state_indices or decode_prefix_len:
            raise ValueError("native PD reservation does not have a prefix or auxiliary state port")
        self.slot = aux_index
        self.call("reserve_handoff", pages=self.pages(kv_indices), metadata_slot=self.slot)

    def poll(self):
        if self.slot is None:
            return KVPoll.WaitingForInput if self.initialized else KVPoll.Bootstrapping
        if self.conclude_state is not None:
            return self.conclude_state

        response = self.call("handoff_status")
        if response["state"] != "ready":
            return KVPoll.Transferring
        self.transport.write_row(self.slot, response["metadata"])
        self.conclude_state = KVPoll.Success
        return self.conclude_state
