"""Run one native pipeline stage with simulated deliveries and device readiness."""

import torch
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

from prologue.integrations.frameworks.sglang.control.scheduler import Inbox, NativeScheduler


class Delivery:
    def __init__(self, peer, identity):
        self.peer, self.identity = peer, identity

    def wait(self):
        self.peer.call("wait_send", delivery=self.identity)
        return True


class StageInbox(Inbox):
    def __init__(self, peer, vocabulary, source):
        super().__init__(peer, vocabulary)
        self.source = source

    def recv_requests(self):
        return self.requests(self.peer.call("receive", channel="requests", source=self.source))


class PipelineScheduler(NativeScheduler):
    def __init__(self, peer, model, domain, stage):
        super().__init__(peer, model, domain, stage)
        self.programs = {}
        self.previous = self.pp_group.prev_rank
        self.following = self.pp_group.next_rank
        if not self.pp_group.is_first_rank:
            self.request_receiver = StageInbox(peer, self.model_config.vocab_size, self.previous)
        self.pp_group.send_tensor_dict = self.send_tensors
        self.pp_group.recv_tensor_dict = self.receive_tensors

    def model_result(self, batch, **kwargs):
        if self.pp_group.is_last_rank:
            return super().model_result(batch, **kwargs)
        return GenerationBatchResult(logits_output=None)

    def submit_batch(self, batch, result, proxy, phase, members):
        inbound = proxy.tensors["program"] if proxy is not None else None
        program = self.peer.call("launch", cut="launch", program=inbound, phase=phase, members=members)
        self.programs[id(batch)] = program
        if not self.pp_group.is_last_rank:
            result.pp_hidden_states_proxy_tensors = PPProxyTensors({"program": program})
        return result

    def _pp_prepare_tensor_dict(self, result, batch):
        payload = super()._pp_prepare_tensor_dict(result, batch)
        payload["program"] = self.programs[id(batch)]
        return payload

    def _pp_prep_batch_result(self, batch, mb_metadata, pp_outputs):
        program = self.programs[id(batch)]
        if program != pp_outputs["program"]:
            raise ValueError("native pipeline delivered the output to a different microbatch")
        self.peer.call("cut", program=program, cut="prepare")
        return super()._pp_prep_batch_result(batch, mb_metadata, pp_outputs)

    def process_batch_result(self, batch, result):
        program = self.programs.pop(id(batch))
        self.peer.call("cut", program=program, cut="process")
        if self.publications is not None:
            self.publications.program = program
        Scheduler.process_batch_result(self, batch, result)
        self.peer.call("result_finished", program=program)

    def _pp_send_pyobj_to_next_stage(self, data, async_send=False):
        payload = [{"id": req.rid, "prompt_tokens": len(req.input_ids),
                    "output_tokens": req.sampling_params.max_new_tokens} for req in data]
        return self.send("requests", self.following, payload, async_send)

    def send_tensors(self, tensor_dict, dst=None, all_gather_group=None, async_send=False):
        peer = self.following if dst is None else self.pp_group.ranks[dst]
        kind = tensor_dict["__msg_type__"]
        payload = {"program": tensor_dict["program"], "kind": kind,
                   "tokens": tensor_dict["next_token_ids"].tolist() if kind == "output" else []}
        return self.send("tensors", peer, payload, async_send)

    def receive_tensors(self, src=None, all_gather_group=None):
        peer = self.previous if src is None else self.pp_group.ranks[src]
        payload = self.peer.call("receive", channel="tensors", source=peer)
        values = {"program": payload["program"], "__msg_type__": payload["kind"]}
        if payload["kind"] == "output":
            values["next_token_ids"] = torch.tensor(payload["tokens"], dtype=torch.int64)
        return values

    def send(self, channel, peer, payload, asynchronous):
        work = Delivery(self.peer, self.peer.call("send", channel=channel, target=peer, payload=payload))
        if asynchronous:
            return [P2PWork(work=work, payload=None)]
        work.wait()
        return []
