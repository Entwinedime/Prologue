"""Observe the native cuts where pipeline stages exchange tensors and readiness."""

from collections import Counter
from functools import partial

from torch import Tensor

from prologue.integrations.frameworks.sglang.observation.parallel import ObservedWork


class PipelineObservation:
    def __init__(self, observation):
        self.observation = observation
        self.calls = observation.calls
        self.reset()

    @property
    def active(self):
        return self.observation.capture_calls

    def reset(self):
        self.window = object()
        self.sequence = Counter()
        self.posts = None
        self.recording = None

    def ports(self, tensors):
        return self.calls.ports(**{key: value for key, value in tensors.items() if isinstance(value, Tensor)})

    def launch(self, original, scheduler, slot, *args):
        if not self.active:
            return original(scheduler, slot, *args)

        with self.calls.scope("pipeline_launch", "Scheduler._pp_launch_batch", "pipeline", {"slot": slot}) as call:
            self.recording = call["id"]
            try:
                result = original(scheduler, slot, *args)
                call["batch"] = self.observation.pending_batches[id(scheduler.cur_batch)]["id"]
                return result
            finally:
                self.recording = None

    def prepare(self, original, scheduler, batch, metadata, outputs):
        if not self.active:
            return original(scheduler, batch, metadata, outputs)

        batch_call = self.observation.pending_batches[id(batch)]
        with self.calls.scope("pipeline_prepare", "Scheduler._pp_prep_batch_result", "pipeline",
                              {"batch": batch_call["id"]}) as call:
            call["inputs"] = self.ports(outputs.tensors)
            result = original(scheduler, batch, metadata, outputs)
            call["outputs"] = self.calls.ports(next_token_ids=result.next_token_ids)

        # SGLang records the result-ready event immediately after returning from preparation.
        self.recording = call["id"]
        return result

    def relay(self, original, scheduler, *args):
        try:
            return original(scheduler, *args)
        finally:
            self.recording = None

    def process(self, original, scheduler, batch, result, batch_call):
        with self.calls.scope("pipeline_process", "Scheduler.process_batch_result", "pipeline",
                              {"batch": batch_call["id"]}) as call:
            call["inputs"] = self.calls.ports(next_token_ids=result.next_token_ids)
            value = original(scheduler, batch, result)
            call["resources"] = self.observation.resources(scheduler)
            return value

    def record(self, original, event, stream=None):
        if self.recording is None or (self.calls.stack and self.calls.stack[-1] != self.recording):
            return original(event, stream)

        with self.calls.scope("signal_record", "Event.record", "pipeline", {"producer": self.recording}) as call:
            result = original(event, stream)
            event._prologue_pipeline_signal = (self.window, call["id"])
            return result

    def wait(self, operation, original, event, *args, **kwargs):
        signal = getattr(event, "_prologue_pipeline_signal", None)
        if not self.active or signal is None or signal[0] is not self.window:
            return original(event, *args, **kwargs)

        with self.calls.scope("signal_wait", f"Event.{operation}", "pipeline",
                              {"producer": signal[1], "host": operation == "synchronize"}):
            return original(event, *args, **kwargs)

    def send(self, original, group, tensor_dict, dst=None, all_gather_group=None, async_send=False):
        return self.message("send", original, group, dst, tensor_dict, all_gather_group, async_send)

    def receive(self, original, group, src=None, all_gather_group=None):
        return self.message("receive", original, group, src, None, all_gather_group)

    def message(self, direction, original, group, peer, sent, *args):
        from sglang.srt.distributed import get_pp_group

        outgoing = direction == "send"
        operands = (sent, peer, *args) if outgoing else (peer, *args)
        if not self.active or group is not get_pp_group():
            return original(group, *operands)

        if peer is None:
            remote = group.next_rank if outgoing else group.prev_rank
        else:
            remote = group.ranks[peer]
        source, target = (group.rank, remote) if outgoing else (remote, group.rank)
        identity = direction, remote
        parameters = {"direction": direction, "source": source, "target": target, "ordinal": self.sequence[identity]}
        self.sequence[identity] += 1
        method = "send_tensor_dict" if outgoing else "recv_tensor_dict"
        with self.calls.scope("pipeline_message", f"GroupCoordinator.{method}", "pipeline", parameters) as call:
            self.posts = []
            try:
                result = original(group, *operands)
                payloads = sent if outgoing else result
                call["inputs" if outgoing else "outputs"] = self.ports(payloads)
                parameters["kind"] = payloads["__msg_type__"]

                # The native transfer reads the nonempty tensor fields in dictionary order.
                fields = [key for key, tensor in payloads.items()
                          if isinstance(tensor, Tensor) and tensor.device.type == "npu" and tensor.numel()]
                for field, post in zip(fields, self.posts, strict=True):
                    post["parameters"]["field"] = field
                return result
            finally:
                self.posts = None

    def post(self, direction, original, tensor, *args, **kwargs):
        if self.posts is None or tensor.device.type != "npu":
            return original(tensor, *args, **kwargs)

        method = "isend" if direction == "send" else "irecv"
        with self.calls.scope("transfer_post", f"distributed.{method}", "pipeline", {"direction": direction}) as call:
            call["inputs" if direction == "send" else "outputs"] = self.calls.ports(wire=tensor)
            work = original(tensor, *args, **kwargs)
            self.posts.append(call)
            return ObservedWork(self.observation, work, "transfer_wait", "pipeline", call["id"])

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        scheduler = "sglang.srt.managers.scheduler.Scheduler."
        group = "sglang.srt.distributed.parallel_state.GroupCoordinator."
        for name, hook in (
            (scheduler + "_pp_launch_batch", self.launch),
            (scheduler + "_pp_prep_batch_result", self.prepare),
            (scheduler + "_pp_send_recv_and_preprocess_output_tensors", self.relay),
            (group + "send_tensor_dict", self.send),
            (group + "recv_tensor_dict", self.receive),
            ("torch.distributed.isend", partial(self.post, "send")),
            ("torch.distributed.irecv", partial(self.post, "receive")),
            ("torch.npu.Event.record", self.record),
            ("torch.npu.Event.wait", partial(self.wait, "wait")),
            ("torch.npu.Event.synchronize", partial(self.wait, "synchronize")),
        ):
            HookRegistry.register(name, hook, HookType.AROUND)
