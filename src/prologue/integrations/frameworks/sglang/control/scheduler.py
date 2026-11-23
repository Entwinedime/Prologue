"""Run SGLang's non-overlapped scheduler with core-owned resource and completion services."""

from array import array
from functools import partial

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import scheduler as native_scheduler
from sglang.srt.managers.io_struct import BatchTokenIDOutput, TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor import model_runner
from sglang.srt.observability import req_time_stats
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs

from prologue.integrations.frameworks.sglang.batch import BatchWork
from prologue.integrations.frameworks.sglang.configuration import scheduling_options
from prologue.integrations.frameworks.sglang.control.resources import PageView
from prologue.integrations.frameworks.sglang.control.runner import MetadataRunner
from prologue.integrations.frameworks.sglang.expert_routes import ExpertRoutes


class Finished(Exception):
    """The core closed a fully drained session."""


class Clock:
    def __init__(self, peer, native):
        self.peer, self.native = peer, native

    def perf_counter(self):
        # Native timestamp fields reserve zero as an unset value.
        return 1.0 + self.peer.now_ns / 1_000_000_000

    monotonic = perf_counter
    time = perf_counter

    def __getattr__(self, name):
        return getattr(self.native, name)


class Inbox:
    def __init__(self, peer, vocabulary, mode=DisaggregationMode.NULL, *, content=None):
        self.peer, self.vocabulary = peer, vocabulary
        self.mode = mode
        self.content = content
        if content is not None and content.vocabulary > vocabulary:
            raise ValueError("the model vocabulary cannot distinguish the declared prompt branches")

    def recv_requests(self):
        turn = self.peer.call("arrivals")
        if turn["done"]:
            raise Finished
        return self.requests(turn["requests"])

    def requests(self, entries):
        requests = []
        for request in entries:
            sampling = SamplingParams(max_new_tokens=request["output_tokens"], ignore_eos=True, temperature=0)
            sampling.normalize(None)
            sampling.verify(self.vocabulary)
            tokens = (array("q", self.content.prompt(request["id"])) if self.content is not None else
                      array("q", [1]) * request["prompt_tokens"])
            requests.append(TokenizedGenerateReqInput(
                rid=request["id"], input_text=None, input_ids=tokens,
                input_embeds=None, mm_inputs=None, token_type_ids=None, sampling_params=sampling,
                return_logprob=False, logprob_start_len=-1, top_logprobs_num=0, token_ids_logprob=None,
                stream=True, time_stats=APIServerReqTimeStats(disagg_mode=self.mode),
            ))
        return requests


class Publications:
    def __init__(self, peer):
        self.peer = peer
        self.program = None

    def send_output(self, output, *args, **kwargs):
        if self.peer is None or not isinstance(output, BatchTokenIDOutput):
            return
        publications = zip(output.rids, output.completion_tokens, output.finished_reasons, strict=True)
        for request, count, finished in publications:
            self.peer.call("publish", request=request, count=count, finished=finished is not None, program=self.program)


class NativeScheduler(Scheduler):
    def __init__(self, peer, model_directory, domain, stage=0, *, expert_routes=None, runtime=None, content=None):
        self.peer = peer
        self.content = content
        config = domain["configuration"]
        args = ServerArgs(
            model_path=str(model_directory), dtype=domain["model"]["dtype"], device="cpu",
            load_format="dummy", skip_tokenizer_init=True, attention_backend="torch_native",
            max_running_requests=config["max_running_requests"], max_total_tokens=config["max_total_tokens"],
            max_prefill_tokens=config["max_prefill_tokens"], page_size=config["page_size"],
            disable_overlap_schedule=True,
            disable_cuda_graph=True, disable_custom_all_reduce=True, stream_interval=1, random_seed=13,
            pp_size=config.get("pp", 1), pp_max_micro_batch_size=config.get("microbatch_requests"),
            **{**scheduling_options(config), **(runtime or {})},
        )

        # Native initialization still constructs the policy, caches and event loop.
        original_runner = model_runner.ModelRunner
        original_watchdog = native_scheduler.create_scheduler_watchdog
        rendezvous = model_directory / f"gloo-{args.disaggregation_mode}"
        model_runner.ModelRunner = partial(MetadataRunner, peer=peer, rendezvous=rendezvous.as_uri(),
                                           logical_tp=config["tp"])
        native_scheduler.create_scheduler_watchdog = self.expiration_owner
        try:
            super().__init__(args, PortArgs.init_new(args), gpu_id=0, tp_rank=0, moe_ep_rank=0,
                             pp_rank=stage, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None)
        finally:
            model_runner.ModelRunner = original_runner
            native_scheduler.create_scheduler_watchdog = original_watchdog

        self.request_receiver = Inbox(peer, self.model_config.vocab_size, self.disaggregation_mode, content=content)
        self.expert_alignment = config["tp"] if config.get("ep", 1) > 1 else 1
        self.experts = None
        if expert_routes is not None:
            self.experts = ExpertRoutes(expert_routes["routes"], expert_routes["requests"], expert_routes["layers"],
                                        self.model_config.hf_config.num_hidden_layers)
        self.publications = None
        if self.output_streamer.send_to_detokenizer.socket is not None:
            self.publications = Publications(peer if self.disaggregation_mode != DisaggregationMode.PREFILL else None)
            self.output_streamer.send_to_detokenizer = self.publications
        self.model_worker.forward_batch_generation = self.model_result
        if config.get("prefix_cache", False):
            from prologue.integrations.frameworks.sglang.control.residency import PrefixResidency

            PrefixResidency(peer, self.tree_cache).install()
        PageView.install()

    @staticmethod
    def expiration_owner(*args, **kwargs):
        # The enclosing task owns the wall-clock watchdog for this blocking process.
        return None

    def model_result(self, batch, **kwargs):
        tokens = (torch.tensor([self.content.outputs[request.rid] for request in batch.reqs], dtype=torch.int64)
                  if self.content is not None else torch.ones(batch.batch_size(), dtype=torch.int64))
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=tokens,
        )

    def run_batch(self, batch, pp_proxy_tensors=None):
        phase = BatchWork.phase(batch.forward_mode)
        members = self.members(batch)
        result = super().run_batch(batch, pp_proxy_tensors)
        return self.submit_batch(batch, result, pp_proxy_tensors, phase, members)

    @staticmethod
    def members(batch):
        return BatchWork.members(batch, (request.rid for request in batch.reqs), slot_offset=1)

    def submit_batch(self, batch, result, proxy, phase, members):
        # Result processing begins only after the core closes the device and host boundaries.
        action = {"phase": phase, "members": members}
        if self.experts is not None:
            action["expert_routes"] = self.experts.aligned(batch, self.expert_alignment)
        self.peer.call("execute", **action)
        return result

    def process_batch_result(self, batch, result):
        super().process_batch_result(batch, result)
        self.peer.call("result_finished")

    def on_idle(self):
        super().on_idle()
        self.peer.call("idle")

    def drive(self):
        native_scheduler.time = Clock(self.peer, native_scheduler.time)
        req_time_stats.time = Clock(self.peer, req_time_stats.time)
        req_time_stats.monotonic_time = req_time_stats.time.monotonic
        try:
            self.run_event_loop()
        except Finished:
            return
