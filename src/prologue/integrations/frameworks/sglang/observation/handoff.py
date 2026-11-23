"""Observe PD ownership and the actual native transport endpoints."""

import json
from contextlib import contextmanager
from functools import partial
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.integrations.frameworks.sglang.observation.pd import PageCopies
from prologue.integrations.frameworks.sglang.page_table import PageTable


class PDHandoff:
    def __init__(self, observation, role, stream):
        self.observation, self.role = observation, role
        self.calls = observation.calls
        self.stream = stream
        self.reset()

    def reset(self):
        self.events = []
        self.rooms = {}
        self.chunks = {}
        self.peers = {}
        self.prepared = {}

    def event(self, kind, **fields):
        row = {"kind": kind, "at_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW), **fields}
        self.events.append(row)
        if self.stream and kind in ("reserve", "commit", "paused", "resumed"):
            with (self.observation.output / "handoff-live.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")

    def identify(self, request):
        identity = self.observation.requests.get(request.rid)
        if identity is not None:
            self.rooms[request.bootstrap_room] = identity
        return identity

    def enqueue(self, original, scheduler, request, last_chunk=False, end_idx=None):
        identity = self.identify(request)
        if identity is None:
            return original(scheduler, request, last_chunk, end_idx)

        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        begin = request.start_send_idx
        result = original(scheduler, request, last_chunk, end_idx)
        self.event("offer", request=identity, room=request.bootstrap_room, slot=request.req_pool_idx,
                   tokens=[begin, request.start_send_idx], last=last_chunk, begin_ns=began,
                   metadata_slot=request.metadata_buffer_index, token=request.output_ids[-1] if last_chunk else None)
        return result

    def chunk(self, original, manager, room, pages, positions, last, *args, **kwargs):
        if room in self.rooms:
            self.chunks[id(pages)] = room
            for peer in manager.transfer_infos[room].values():
                registration = manager.decode_kv_args_table[peer.mooncake_session_id]
                self.peers[room, peer.endpoint, peer.dst_port] = registration.dst_tp_rank
            self.event("source_pages", request=self.rooms[room], room=room, pages=pages.tolist(),
                       positions=[positions.start, positions.stop])
        return original(manager, room, pages, positions, last, *args, **kwargs)

    def payload(self, sliced, original, manager, peer, source, pointers, target, *args):
        room = self.chunks.get(id(source))
        if room is None:
            return original(manager, peer, source, pointers, target, *args)

        # The different-width call receives the actual Decode lane and its registered row layout.
        registration = manager.decode_kv_args_table[peer]
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(manager, peer, source, pointers, target, *args)
        self.event("kv", request=self.rooms[room], room=room, peer=registration.dst_tp_rank,
                   source_pages=source.tolist(), target_pages=target.tolist(), layers=len(pointers) // 2,
                   bytes_per_page=registration.dst_kv_item_len, sliced=sliced, begin_ns=began, status=result)
        return result

    def metadata(self, original, manager, request, source, destination):
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(manager, request, source, destination)
        if request.room in self.rooms:
            peer = manager.decode_kv_args_table[request.mooncake_session_id].dst_tp_rank
            self.event("metadata", request=self.rooms[request.room], room=request.room, peer=peer,
                       source_slot=source, target_slot=request.dst_aux_index, bytes=sum(manager.kv_args.aux_item_lens),
                       begin_ns=began, status=result)
        return result

    def notify(self, original, manager, remote, port, room, status, prefill_rank):
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(manager, remote, port, room, status, prefill_rank)
        if room in self.rooms:
            self.event("notify", request=self.rooms[room], room=room, peer=self.peers[room, remote, port],
                       status=int(status), begin_ns=began)
        return result

    def reserve(self, original, queue, request, *args, **kwargs):
        identity = self.identify(request)
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        loc = original(queue, request, *args, **kwargs)
        if identity is not None:
            page_size = queue.token_to_kv_pool_allocator.page_size
            pages = loc[::page_size].detach().cpu().tolist()
            self.event("reserve", request=identity, room=request.bootstrap_room, slot=request.req_pool_idx,
                       pages=[page // page_size for page in pages], tokens=request.kv_committed_len,
                       begin_ns=began)
        return loc

    def ready(self, original, manager, room, status):
        from sglang.srt.disaggregation.base import KVPoll

        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(manager, room, status)
        if room in self.rooms and status == KVPoll.Success:
            self.event("ready", request=self.rooms[room], room=room, begin_ns=began)
        return result

    def commit(self, original, queue, transfer):
        request = transfer.req
        identity = self.observation.requests.get(request.rid)
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        extent = len(request.output_ids)
        result = original(queue, transfer)
        if identity is not None:
            self.event("commit", request=identity, room=request.bootstrap_room, slot=request.req_pool_idx,
                       tokens=request.kv_committed_len, metadata_slot=transfer.metadata_buffer_index,
                       token=request.output_ids[-1] if len(request.output_ids) == extent + 1 else None, begin_ns=began)
        return result

    @contextmanager
    def copy(self, row, operand, output):
        if not self.observation.capture_calls:
            yield
            return

        query = PageTable.copy(row)
        scope = self.calls.scope(PageTable.copy_operation, "sglang.prebuilt.page_table_assignment", "decode", query)
        with scope as call:
            call["inputs"] = self.calls.ports(source=operand)
            yield
            call["outputs"] = self.calls.ports(destination=output[slice(*row["target"])])
            row["operand"] = call["inputs"][0]
            row["view"] = call["outputs"][0]["tensor"]

    def prepare(self, original, batch):
        if not any(request.rid in self.observation.requests for request in batch.reqs):
            return original(batch)

        copies = PageCopies(batch, self.copy)
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        call = None
        if self.observation.capture_calls:
            operation = PageTable.assembly_operation
            with self.calls.scope(operation, "ScheduleBatch.prepare_for_prebuilt", "decode", {}) as call, copies:
                call["resources"] = self.observation.resources(batch)
                result = original(batch)
                call["query"] = PageTable.assembly(copies.records)
                call["outputs"] = self.calls.ports(out_cache_loc=batch.out_cache_loc)
                call["outputs"][0]["views"] = [{"tensor": row.pop("view"), "interval": row["target"]}
                                                for row in copies.records]
                call["inputs"] = [{**row.pop("operand"), "name": f"member_{index}_pages"}
                                  for index, row in enumerate(copies.records)]
        else:
            with copies:
                result = original(batch)

        members = [
            {"request": self.observation.requests[request.rid], "slot": request.req_pool_idx,
             "tokens": request.already_computed, "copy": row}
            for request, row in zip(batch.reqs, copies.records, strict=True)
        ]
        if call is not None:
            self.prepared[id(batch)] = call
        self.event("admit", begin_ns=began, members=members, extent=batch.out_cache_loc.numel(),
                   writes_output=all(value is batch.out_cache_loc for value in copies.targets))
        return result

    def finish_prebuilt(self, original, processor, batch):
        call = self.prepared.pop(id(batch), None)
        if call is None:
            return original(processor, batch)
        began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(processor, batch)
        call["result_processing"] = {"begin_ns": began, "end_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
                                     "resources": self.observation.resources(processor)}
        return result

    def admission_switch(self, kind, original, scheduler, request):
        result = original(scheduler, request)
        self.event(kind)
        return result

    def read(self, original, backend, *args, **kwargs):
        if self.calls.stack:
            root = self.calls.rows[self.calls.stack[0]]
            if "cache" in root and "blocks" not in root["cache"]:
                with self.calls.scope("capture_auxiliary", "BlockTable.snapshot", "decode", {}):
                    root["cache"]["blocks"] = backend.forward_metadata.block_tables.detach().cpu().tolist()
        return original(backend, *args, **kwargs)

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        mooncake = "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager."
        if self.role == "prefill":
            hooks = (
                ("sglang.srt.managers.scheduler.Scheduler.send_kv_chunk", self.enqueue),
                (mooncake + "add_transfer_request", self.chunk),
                (mooncake + "send_kvcache_slice", partial(self.payload, True)),
                ("sglang.srt.disaggregation.ascend.conn.AscendKVManager.send_kvcache", partial(self.payload, False)),
                (mooncake + "send_aux", self.metadata),
                (mooncake + "sync_status_to_decode_endpoint", self.notify),
            )
        else:
            decode = "sglang.srt.disaggregation.decode."
            attention = "sglang.srt.hardware_backend.npu.attention.ascend_backend.AscendAttnBackend."
            scheduler = "sglang.srt.managers.scheduler.Scheduler."
            hooks = (
                (decode + "DecodePreallocQueue._pre_alloc", self.reserve),
                (decode + "DecodeTransferQueue._commit_transfer_to_req", self.commit),
                ("sglang.srt.disaggregation.common.conn.CommonKVManager.update_status", self.ready),
                ("sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_prebuilt", self.prepare),
                ("sglang.srt.managers.scheduler_components.batch_result_processor."
                 "SchedulerBatchResultProcessor.process_batch_result_prebuilt", self.finish_prebuilt),
                (scheduler + "pause_generation", partial(self.admission_switch, "paused")),
                (scheduler + "continue_generation", partial(self.admission_switch, "resumed")),
                (attention + "forward_decode", self.read),
            )
        for name, observer in hooks:
            HookRegistry.register(name, observer, HookType.AROUND)
