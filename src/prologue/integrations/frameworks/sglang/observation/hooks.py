"""Translate native serving calls into the shared observation contract."""

import json
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns, thread_time_ns

from prologue.acquisition.observation.calls import Calls
from prologue.integrations.frameworks.sglang.batch import BatchWork
from prologue.integrations.frameworks.sglang.expert_routes import ExpertRoutes
from prologue.integrations.frameworks.sglang.observation.attention import AttentionCall, implementation
from prologue.integrations.frameworks.sglang.observation.parallel import RankProfiles


class Observation:
    def __init__(self, output, document):
        requests, mode = document["requests"], document["mode"]
        parallel = document["parallel_placement"] is not None
        self.output = output
        self.requests = requests
        self.publications = {native: [] for native in requests}
        self.finished = set()
        self.mode = mode
        self.capture_calls = mode == "detailed" and not parallel
        self.parallel = RankProfiles(self) if parallel else None
        self.pending_batches = {}
        self.result_batch = None
        self.result_cpu_started = None
        self.host_work = []
        self.prepare_cpu_ns = 0
        self.measure_arrivals = mode == "detailed" and document["pp"] == 1
        self.snapshot_cache = mode == "detailed" and document.get("snapshot_cache", False)
        self.ingress = {}
        self.awaiting_read = None
        self.sampling = document.get("synthetic_sampling")
        self.batch_tokens = None
        self.calls = None
        if mode == "detailed":
            from torch.autograd.profiler import record_function

            self.calls = Calls(record_function)
        self.pipeline = None
        if mode == "detailed" and document["pp"] > 1:
            from prologue.integrations.frameworks.sglang.observation.pipeline import PipelineObservation

            self.pipeline = PipelineObservation(self)
        self.experts = None
        if experts := document.get("experts"):
            from prologue.integrations.frameworks.sglang.observation.experts import ExpertObservation

            self.experts = ExpertObservation(self, experts)
        self.pd = None
        if mode == "detailed" and (disaggregation := document.get("disaggregation")):
            from prologue.integrations.frameworks.sglang.observation.handoff import PDHandoff

            self.pd = PDHandoff(self, disaggregation, document.get("stream_handoff", False))
        self.cache = None
        if document.get("prefix_cache", False) and self.snapshot_cache:
            from prologue.integrations.frameworks.sglang.observation.residency import CacheObservation

            self.cache = CacheObservation(self)

    def begin_window(self, output, requests, capture_calls):
        self.output = output
        self.requests = requests
        self.publications = {native: [] for native in requests}
        self.finished.clear()
        self.ingress.clear()
        self.awaiting_read = None
        self.capture_calls = capture_calls
        self.calls.reset()
        if self.cache is not None:
            self.cache.reset()
        if self.pipeline is not None:
            self.pipeline.reset()
        if self.pd is not None:
            self.pd.reset()

    def flush(self, original, scheduler, empty_cache=True):
        if self.cache is not None:
            self.cache.bind(scheduler.tree_cache)

        # Drain logical KV state while preserving warmed device allocator workspaces.
        drained = original(scheduler, empty_cache=False)
        if self.parallel is None and drained and self.finished == self.publications.keys():
            self.save()
        return drained

    def publish(self, original, sender, output, *args, **kwargs):
        from sglang.srt.managers.io_struct import BatchTokenIDOutput

        result = original(sender, output, *args, **kwargs)
        if sender.socket is None or not isinstance(output, BatchTokenIDOutput):
            return result

        now = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        for rid, count, finished in zip(output.rids, output.completion_tokens, output.finished_reasons, strict=True):
            if rid not in self.publications or rid in self.finished:
                continue
            rows = self.publications[rid]
            if count > (rows[-1]["count"] if rows else 0):
                rows.append({"count": count, "raw_ns": now})
                if self.result_cpu_started is not None:
                    elapsed = thread_time_ns() - self.result_cpu_started
                    self.result_batch["publication_cpu_ns"][self.requests[rid]] = elapsed
            if finished is not None:
                self.finished.add(rid)
        return result

    def save(self):
        document = {"publications": {self.requests[rid]: rows for rid, rows in self.publications.items()}}
        if self.capture_calls:
            if self.pending_batches:
                raise ValueError("the native observation closed before every captured batch finished")
            document["calls"] = self.calls.rows
            if self.measure_arrivals:
                document["ingress"] = self.ingress
            if self.cache is not None:
                document["cache_effects"] = self.cache.effects
        elif self.mode == "host-timing":
            document["host_work"] = self.host_work
        if self.pd is not None:
            document["pd_events"] = self.pd.events

        # Each selected worker writes its own calls. Only the publication owner has token records.
        self.output.mkdir(exist_ok=True)
        path = self.output / "worker.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
        pending.replace(path)

    def prepare(self, original, scheduler, *args, **kwargs):
        began = thread_time_ns()
        batch = original(scheduler, *args, **kwargs)
        self.prepare_cpu_ns = thread_time_ns() - began
        return batch

    def receive(self, generate_batch, original, receiver):
        inputs = original(receiver)
        if not self.capture_calls:
            return inputs
        read = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        if self.awaiting_read is not None:
            self.awaiting_read["next_read_ns"] = read
            self.awaiting_read = None
        for item in inputs:
            requests = item.batch if isinstance(item, generate_batch) else (item,)
            for request in requests:
                identity = self.requests.get(getattr(request, "rid", None))
                if identity is not None:
                    self.ingress[identity] = read
        return inputs

    def batch(self, original, scheduler, batch, pp_proxy_tensors=None):
        if self.sampling is None:
            return self.observe_batch(original, scheduler, batch, pp_proxy_tensors)
        outputs, warmup = self.sampling["outputs"], self.sampling["warmup"]
        self.batch_tokens = [outputs[self.requests.get(request.rid) or warmup[request.rid]] for request in batch.reqs]
        try:
            return self.observe_batch(original, scheduler, batch, pp_proxy_tensors)
        finally:
            self.batch_tokens = None

    def sample(self, original, runner, *args, **kwargs):
        tokens = original(runner, *args, **kwargs)
        if self.batch_tokens is not None:

            # Preserve the native sampler and tensor identity while making later cache keys reproducible.
            scope = (self.calls.scope("capture_auxiliary", "ModelRunner.synthetic_output", "sampling", {})
                     if self.calls is not None and self.calls.stack else nullcontext())
            with scope:
                tokens.copy_(tokens.new_tensor(self.batch_tokens))
        return tokens

    def observe_batch(self, original, scheduler, batch, pp_proxy_tensors):
        if self.mode == "light":
            return original(scheduler, batch, pp_proxy_tensors)
        if self.mode == "detailed" and not self.capture_calls:
            return original(scheduler, batch, pp_proxy_tensors)

        requests = [self.requests.get(request.rid) for request in batch.reqs]
        if not any(request is not None for request in requests):
            return original(scheduler, batch, pp_proxy_tensors)
        if any(request is None for request in requests):
            raise ValueError("warmup and observed requests occupied the same batch")

        phase = BatchWork.phase(batch.forward_mode)
        members = BatchWork.members(batch, requests)
        if self.mode == "host-timing":
            call = {"phase": phase, "members": members, "prepare_cpu_ns": self.prepare_cpu_ns,
                    "publication_cpu_ns": {}}
            self.pending_batches[id(batch)] = call
            began = thread_time_ns()
            value = original(scheduler, batch, pp_proxy_tensors)
            call["forward_cpu_ns"] = thread_time_ns() - began
            return value

        parameters = {"phase": phase, "members": len(requests)}
        with self.calls.scope("batch", "Scheduler.run_batch", "scheduler", parameters) as call:
            if self.cache is not None:
                self.cache.actions += 1
            self.pending_batches[id(batch)] = call
            call["members"] = members
            call["resources"] = self.resources(scheduler)
            if self.pd is not None or self.snapshot_cache:
                with self.calls.scope("capture_auxiliary", "PageTable.snapshot", phase, {}):
                    call["cache"] = BatchWork.cache(batch, members, prefix_cache=self.cache is not None)
            if pp_proxy_tensors is not None:
                call["inputs"] = self.pipeline.ports(pp_proxy_tensors.tensors)
            result = original(scheduler, batch, pp_proxy_tensors)
            if self.experts is not None:
                call["expert_routes"] = self.experts.bindings
            if self.pipeline is not None:
                proxy = result.pp_hidden_states_proxy_tensors
                call["outputs"] = (self.pipeline.ports(proxy.tensors) if proxy is not None else
                                   self.calls.ports(next_token_ids=result.next_token_ids))
            return result

    def resources(self, scheduler):
        capacity = {
            "slots": scheduler.req_to_token_pool.available_size(),
            "kv_tokens": scheduler.token_to_kv_pool_allocator.available_size(),
        }
        if self.cache is not None:
            capacity.update(evictable_tokens=self.cache.tree.evictable_size(),
                            protected_tokens=self.cache.tree.protected_size())
        return capacity

    def process_result(self, original, scheduler, batch, result):
        call = self.pending_batches.pop(id(batch), None)
        if call is None:
            return original(scheduler, batch, result)

        if self.mode == "host-timing":
            self.result_batch = call
            self.result_cpu_started = thread_time_ns()
            try:
                value = original(scheduler, batch, result)
                call["result_cpu_ns"] = thread_time_ns() - self.result_cpu_started
                self.host_work.append(call)
                return value
            finally:
                self.result_cpu_started = None
                self.result_batch = None

        if self.pipeline is not None:
            return self.pipeline.process(original, scheduler, batch, result, call)

        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        value = original(scheduler, batch, result)
        self.processed(call, scheduler, begin)
        return value

    def processed(self, call, scheduler, begin):
        call["result_processing"] = {
            "begin_ns": begin, "end_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
            "resources": self.resources(scheduler),
        }
        if self.measure_arrivals:
            self.awaiting_read = call["result_processing"]

    def layer(self, original, module, site, positions, hidden_states, forward_batch, residual, *args, **kwargs):
        if not self.calls.stack:
            return original(positions, hidden_states, forward_batch, residual, *args, **kwargs)

        with self.calls.scope("layer", implementation(module), site, {}) as call:
            call["inputs"] = self.calls.ports(positions=positions, hidden=hidden_states, residual=residual)
            result = original(positions, hidden_states, forward_batch, residual, *args, **kwargs)
            call["outputs"] = self.calls.ports(hidden=result[0], residual=result[1])
            return result

    def projection(self, original, module, site, input_, *args, **kwargs):
        if not self.calls.stack:
            return original(input_, *args, **kwargs)

        with self.calls.scope("projection", implementation(module), site, {}) as call:
            call["inputs"] = self.calls.ports(input=input_)
            result = original(input_, *args, **kwargs)
            call["outputs"] = self.calls.ports(output=result[0], bias=result[1])
            return result

    def load_model(self, original, runner):
        from sglang.srt.layers.radix_attention import RadixAttention

        result = original(runner)
        if self.sampling is not None and self.sampling["vocabulary"] > runner.model_config.vocab_size:
            raise ValueError("the model vocabulary cannot distinguish the declared prompt branches")
        routed_modules = ExpertRoutes.native_modules(runner.model)
        if bool(routed_modules) != (self.experts is not None):
            raise ValueError("a workload expert route table is required exactly for models with native routed experts")
        if self.experts is not None:
            self.experts.loaded(runner, routed_modules)
        if self.mode != "detailed":
            return result

        body = runner.model.model
        for index in range(body.start_layer, body.end_layer):
            layer = body.layers[index]
            site = f"layers.{index}"
            layer.forward = partial(self.layer, layer.forward, layer, site)
            self_attention = layer.self_attn
            attention = self_attention.attn
            if not isinstance(attention, RadixAttention):
                raise ValueError(f"unsupported attention implementation at {site}")

            # Separate the producer and its native consumer so the compiler can retain their tensor edge.
            projection = self_attention.o_proj
            attention.forward = AttentionCall(self.calls, attention, f"{site}.self_attn.attn", require_parent=True)
            projection.forward = partial(self.projection, projection.forward, projection, f"{site}.self_attn.o_proj")
        if self.parallel is not None:
            self.parallel.loaded(runner)
        return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        scheduler = "sglang.srt.managers.scheduler.Scheduler."
        sender = "sglang.srt.managers.scheduler_components.output_sender.SenderWrapper.send_output"
        HookRegistry.register(scheduler + "flush_cache", self.flush, HookType.AROUND)
        HookRegistry.register(sender, self.publish, HookType.AROUND)
        if self.mode != "light" or self.sampling is not None:
            HookRegistry.register(scheduler + "run_batch", self.batch, HookType.AROUND)
        if self.mode != "light":
            HookRegistry.register(scheduler + "process_batch_result", self.process_result, HookType.AROUND)
        if self.sampling is not None:
            HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.sample",
                                  self.sample, HookType.AROUND)
        if self.mode == "host-timing":
            HookRegistry.register(scheduler + "get_next_batch_to_run", self.prepare, HookType.AROUND)
        if self.measure_arrivals:
            from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput

            receiver = "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver."
            HookRegistry.register(receiver + "recv_requests", partial(self.receive, BatchTokenizedGenerateReqInput),
                                  HookType.AROUND)
        HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.load_model",
                              self.load_model, HookType.AROUND)
        if self.parallel is not None:
            self.parallel.install()
        if self.pipeline is not None:
            self.pipeline.install()
        if self.experts is not None:
            self.experts.install()
        if self.pd is not None:
            self.pd.install()
        if self.cache is not None:
            self.cache.install()


def install():
    pd_cut = os.environ.get("PROLOGUE_PD_CUT")
    if pd_cut is not None:
        from prologue.integrations.frameworks.sglang.observation.pd import PDObservation

        specification = Path(pd_cut)
        PDObservation(specification.parent, json.loads(specification.read_text(encoding="utf-8"))).install()
        return

    path = os.environ.get("PROLOGUE_OBSERVATION")
    if path is not None:
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        placement = document["parallel_placement"]
        pipeline = document["pp"] > 1
        experts = document.get("experts")
        if placement == "single_device":
            from prologue.integrations.frameworks.sglang.ascend.shards import StagedTransport

            StagedTransport(Path(document.get("device_lock", path.parent / ".device.lock"))).install(
                pipeline, experts is not None and experts["width"] > 1,
            )
        Observation(path.parent, document).install()
