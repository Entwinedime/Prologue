"""Drive one of the native Prefill and Decode event loops."""

from sglang.srt.disaggregation import decode
from sglang.srt.managers.io_struct import ContinueGenerationReqInput, PauseGenerationReqInput
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from prologue.integrations.frameworks.sglang.configuration import Disaggregation
from prologue.integrations.frameworks.sglang.control.scheduler import Clock, Inbox, NativeScheduler
from prologue.integrations.frameworks.sglang.control.transport import Transport
from prologue.integrations.frameworks.sglang.page_table import PageTable


class PDInbox(Inbox):
    def __init__(self, peer, vocabulary, mode, transport):
        super().__init__(peer, vocabulary, mode)
        self.transport = transport

    def recv_requests(self):
        turn = self.peer.call("arrivals")
        requests = self.requests(turn["requests"])
        for command in turn["admission"]:
            if command == "pause":
                requests.append(PauseGenerationReqInput(mode="in_place"))
            else:
                requests.append(ContinueGenerationReqInput(torch_empty_cache=False))
        return requests

    def requests(self, entries):
        requests = super().requests(entries)
        for request in requests:
            request.bootstrap_host = "prologue-control"
            request.bootstrap_room = self.transport.room(request.rid)
        return requests


class DisaggregatedScheduler(NativeScheduler):
    def __init__(self, peer, model, domain, role):
        pair = Disaggregation(**domain["configuration"])
        self.role = role
        self.transport = Transport(peer, pair)
        config = pair.instances[role].document()
        with self.transport.install():
            super().__init__(peer, model, {**domain, "configuration": config}, runtime=pair.native_policy(role))
        self.transport.buffers = self.disagg_metadata_buffers
        self.request_receiver = PDInbox(peer, self.model_config.vocab_size, self.disaggregation_mode, self.transport)
        if role == "decode":
            self.install_decode_boundaries()

    def install_decode_boundaries(self):
        hooks = (
            ("sglang.srt.disaggregation.decode.DecodePreallocQueue._pre_alloc", self.preallocate),
            ("sglang.srt.managers.scheduler_components.batch_result_processor."
             "SchedulerBatchResultProcessor.process_batch_result_prebuilt", self.finish_prebuilt),
        )
        for target, function in hooks:
            HookRegistry.register(target, function, HookType.AROUND)
        HookRegistry.apply_hooks()

    @staticmethod
    def preallocate(original, queue, request, *args, **kwargs):
        with queue.token_to_kv_pool_allocator.binding((request,)):
            return original(queue, request, *args, **kwargs)

    def get_new_prebuilt_batch(self):
        batch = super().get_new_prebuilt_batch()
        if batch is None:
            return None

        # CPU metadata executes the native copy. Its ports describe the target Ascend device.
        query = PageTable.assembly([
            {"source": [begin, begin + length],
             "source_dtype": str(batch.req_to_token_pool.req_to_token.dtype),
             "target_dtype": str(batch.out_cache_loc.dtype),
             "source_device": "npu", "target_device": "npu"}
            for begin, length in zip(batch.prefix_lens, batch.extend_lens, strict=True)
        ])
        self.peer.call("admit_handoff", phase=query["phase"], members=self.members(batch), query=query)
        return batch

    def finish_prebuilt(self, original, processor, batch):
        result = original(processor, batch)
        self.peer.call("result_finished")
        return result

    def process_decode_queue(self):
        super().process_decode_queue()
        if self._engine_paused and self.polling_count == 0:
            # Paused native Decode polls transfers and skips its normal idle callback.
            self.peer.call("idle")

    def on_idle(self):
        if self.role == "decode" and self.polling_count and (
            self.disagg_decode_prealloc_queue.queue or self.disagg_decode_transfer_queue.queue
        ):
            # Finish one native poll cycle at this instant before sleeping for another delivery.
            return
        super().on_idle()

    def drive(self):
        native_time = decode.time
        with self.transport.install():
            decode.time = Clock(self.peer, native_time)
            try:
                super().drive()
            finally:
                decode.time = native_time
