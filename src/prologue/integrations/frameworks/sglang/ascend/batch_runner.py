"""Run one requested host or model cut on an initialized native scheduler."""

from contextlib import ExitStack, contextmanager, nullcontext

import torch
import zmq
from sglang.srt.managers.io_struct import BatchTokenIDOutput, sock_recv
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.utils.common import DynamicGradMode

from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.batches import BatchTensors
from prologue.integrations.frameworks.sglang.ingress import RequestSubmissions
from prologue.integrations.frameworks.sglang.observation.communications import NativeCommunications
from prologue.integrations.frameworks.sglang.observation.flow_scopes import FlowScopes
from prologue.integrations.frameworks.sglang.observation.host_samples import NativeHostSamples
from prologue.integrations.frameworks.sglang.observation.model_calls import NativeModelObservation
from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes
from prologue.integrations.frameworks.sglang.observation.source_scopes import SGLangSourceScopes
from prologue.integrations.platforms.ascend.export import AscendExport
from prologue.integrations.platforms.ascend.host_profile import AscendHostProfile
from prologue.integrations.platforms.ascend.markers import AscendMarkers
from prologue.integrations.platforms.ascend.observation import AscendObservation
from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks


class SchedulerBatchObservation(NativeModelObservation):
    def __init__(self, output, description, *, model):
        super().__init__(AscendMarkers(detailed_timing=False), "null")
        self.output, self.description, self.model = output, description, model
        self.aliases, self.routing = {}, None
        self.calls = FlowScopes(self)
        self.communications = NativeCommunications(self)

    def install(self):
        if self.model:
            ModelScopes.install_batch_hooks(self)
            if self.description["worker_count"] > 1:
                self.communications.install()
        else:
            self.calls.install_calls()
        self.request_state.install()

    def staging(self):
        return self.tensors.auxiliary, self.communications.local_exchange

    def bind(self, scheduler, layout, env):
        self.scheduler, self.environment = scheduler, env
        self.parallel, self.groups, self.worker_roles = layout.parallel, layout.groups, env["worker_roles"]
        self.tensors.namespace = f"rank/{self.parallel['rank']}/"
        self.kv_layout = layout.kv_layout
        if self.model:
            ModelScopes(self).attach_model(layout.model)
        else:
            self.calls.bind(scheduler)

    @contextmanager
    def recording(self):
        runtime = self.capture.runtime
        runtime.synchronize()
        with ScopedTasks.profile(self.output, self.parallel["rank"]), runtime.window():
            self.active = True
            try:
                yield
            finally:
                self.active = False
        runtime.synchronize()

    def export(self):
        rank = self.parallel["rank"]
        runtime = self.capture.runtime
        self.communications.bind_returned_operands()
        raw = self.capture.document(self.environment)
        raw.update(runtime.timing_evidence(), state_queries=self.calls.queries)
        write_json(self.output / f"rank-{rank}.json", raw)
        runtime.release()
        sdk = self.output / "sdk"
        AscendExport.run(self.output, rank, sdk)
        raw["scopes"] = SGLangSourceScopes(self.environment).lower(raw["scopes"])
        facts = AscendObservation(self.output, rank, raw, sdk / "trace.json", evidence=False).document()
        write_json(self.output / "facts.json", facts)


class SchedulerBatchHostTiming:
    def __init__(self, output, description, *, model):
        self.output, self.description, self.model = output, description, model
        self.aliases = {}
        self.calls = NativeHostSamples(self.aliases, description, AscendHostProfile)

    def install(self):
        if self.model:
            ModelScopes.install_batch_hooks(self.calls)
            if self.description["worker_count"] > 1:
                self.calls.communications.install()
        else:
            self.calls.install_calls()
        self.calls.request_state.install()

    def staging(self):
        return self.calls.auxiliary, self.calls.local_exchange

    def bind(self, scheduler, layout, env):
        self.scheduler = self.calls.scheduler = scheduler
        self.rank, self.environment = layout.parallel["rank"], env
        if self.model:
            self.calls.attach_model(layout.model)
        else:
            self.calls.bind(scheduler)

    @contextmanager
    def recording(self):
        self.calls.start(self.scheduler, self.rank, self.output)
        self.calls.enabled = True
        try:
            yield
        finally:
            self.calls.enabled = False

    def export(self):
        write_json(self.output / "host.json", self.calls.document() | {"rank": self.rank, "environment": self.environment})


class SchedulerBatch:
    def __init__(self, scheduler, observer, *, model, observed):
        self.scheduler, self.observer = scheduler, observer
        self.aliases, self.model_cut = observer.aliases, model
        self.record = observer.recording if observed else nullcontext
        self.publishes = scheduler.ipc_channels.send_to_detokenizer.socket is not None
        self.preparation = None

    def receive(self, receiver):
        if not self.publishes:
            return []
        if not receiver.poll(5000):
            raise ValueError("native result did not publish a token message to the local receiver")
        message = sock_recv(receiver)
        if not isinstance(message, BatchTokenIDOutput):
            raise ValueError("native output channel returned a different message type")
        return [{"id": rid, "tokens": list(ids), "generated_tokens": count, "finished": reason is not None}
                for rid, ids, count, reason in zip(message.rids, message.output_ids, message.completion_tokens,
                                                   message.finished_reasons, strict=True)]

    def prime_batch(self, tokens, receiver):
        batch = self.scheduler.get_next_batch_to_run()
        if batch is None:
            raise ValueError("the native scheduler ran out of work while preparing local request progress")
        self.advance(batch, tokens, receiver)

    def advance(self, batch, tokens, receiver):
        scheduler = self.scheduler
        scheduler.cur_batch = batch
        supplied = torch.tensor([tokens[req.rid] for req in batch.reqs], dtype=torch.int64, device=scheduler.device)
        worker = scheduler.model_worker
        forward = worker.forward_batch_generation

        def supply(batch, **kwargs):
            if not self.model_cut:
                return GenerationBatchResult(next_token_ids=supplied)

            # Future model work reads real KV. Let the native forward fill it before publishing the input token.
            output = forward(batch, **kwargs)
            output.next_token_ids.copy_(supplied)
            return output

        try:
            worker.forward_batch_generation = supply
            scheduler.process_batch_result(batch, scheduler.run_batch(batch))
        finally:
            worker.forward_batch_generation = forward
        scheduler.last_batch = batch
        return self.receive(receiver)

    def operands(self, rows, output_tokens=None):
        scheduler = self.scheduler
        requests, tokens = {}, {}
        for row in rows:
            identity = row["id"]
            prompt = (token for token, length in row["prompt"] for _ in range(length))
            requests[identity] = RequestSubmissions.tokenized(
                identity, prompt, {"max_new_tokens": row["output_tokens"] if output_tokens is None else output_tokens,
                                   "temperature": 0, "ignore_eos": True},
                scheduler.model_config.vocab_size, scheduler.disaggregation_mode,
            )
            tokens[identity] = row["next_token"]
            self.aliases[identity] = identity
        return requests, tokens

    def warm_cost(self, rows, receiver):
        scheduler = self.scheduler
        allocator, slots = scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
        free = allocator.available_size(), slots.available_size()

        # Exercise the actual prompt shapes before timing, with just one Prefill and one Decode per request.
        requests, tokens = self.operands(rows, output_tokens=2)
        scheduler.process_input_requests(list(requests.values()))
        finished, batches = set(), 0
        while (batch := scheduler.get_next_batch_to_run()) is not None:
            published = self.advance(batch, tokens, receiver)
            finished.update(row["id"] for row in published if row["finished"])
            batches += 1
        scheduler.cur_batch = scheduler.last_batch = None
        torch.npu.synchronize()
        if self.publishes and finished != requests.keys():
            raise ValueError("native warmup did not complete every local request")
        if not scheduler.flush_cache(empty_cache=False) or (allocator.available_size(), slots.available_size()) != free:
            raise ValueError("native warmup did not restore the original free slots and KV capacity")
        self.preparation = {"requests": len(requests), "native_batches": batches, "free_tokens": free[0], "free_slots": free[1]}

    def prepare(self, inputs, receiver):
        scheduler = self.scheduler
        requests, tokens = self.operands(inputs["requests"])

        for cohort in inputs["priming"]:
            scheduler.process_input_requests([requests[identity] for identity in cohort["admit"]])
            self.prime_batch(tokens, receiver)
            for _ in range(cohort["decode_steps"]):
                self.prime_batch(tokens, receiver)
        if inputs["phase"] == "scheduler" and inputs["waiting"]:
            scheduler.process_input_requests([requests[rid] for rid in inputs["waiting"]])

        result = None
        if not self.model_cut and inputs["phase"] in {"prefill", "decode"}:
            result = GenerationBatchResult(next_token_ids=torch.tensor(list(tokens.values()), dtype=torch.int64, device=scheduler.device))
        return list(requests.values()), result

    def select(self, phase, ingress):
        scheduler = self.scheduler
        if phase in {"admission", "prefill"}:
            scheduler.process_input_requests(ingress)
        if phase == "admission":
            return None

        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if phase == "idle":
            if batch is not None:
                raise ValueError("native worker selected a batch after its idle-pool flush")
        elif phase not in {"selection", "scheduler"} and (
                batch is None or [req.rid for req in batch.reqs] != list(self.aliases) or BatchTensors.phase(batch) != phase):
            raise ValueError("native selection differs from the requested phase or ordered members")
        return batch

    def observe_host(self, phase, ingress, result, receiver):
        with self.record():
            if phase == "scheduler":
                queues = self.observer.calls.scheduler_work()["context"]
                batch = self.select(phase, ingress)
                return {"phase": phase, "queues": queues, "selected": [req.rid for req in batch.reqs] if batch else []}
            batch = self.select(phase, ingress)
            if phase == "admission":
                return {"phase": phase, "waiting": [req.rid for req in self.scheduler.waiting_queue]}
            if phase == "idle":
                return {"phase": phase, "members": []}
            if phase == "selection":
                return {"phase": phase, "selected": [req.rid for req in batch.reqs] if batch is not None else []}

            members = BatchTensors.members(batch)
            self.scheduler.process_batch_result(batch, result)
        return {"phase": phase, "members": members, "effects": self.observer.calls.completed(batch.reqs),
                "publication": self.receive(receiver)}

    def observe_model(self, phase, ingress):
        scheduler = self.scheduler
        batch = self.select(phase, ingress)
        members = BatchTensors.members(batch)
        with self.record():
            output = scheduler.run_batch(batch)

        # Both adapter checks run after profiling. The next native iteration reads its tokens from these request slots.
        tokens = output.next_token_ids
        relay = scheduler.future_map.output_tokens_buf[batch.req_pool_indices]
        return {"phase": phase, "members": members, "output": {"tokens": tokens.tolist(), "relayed": relay.tolist(),
                "dtype": str(tokens.dtype), "device": tokens.device.type}}

    def observe(self, ports, batch_input=None, *, warm_cost=False):
        scheduler = self.scheduler
        with ExitStack() as stack:
            stack.enter_context(DynamicGradMode())
            scheduler.schedule_stream = scheduler.device_module.Stream(priority=0)
            stack.enter_context(scheduler.device_module.StreamContext(scheduler.schedule_stream))
            phase, ingress, result, receiver = "idle", [], None, None
            if batch_input is not None:
                phase = batch_input["phase"]
                warm = warm_cost and batch_input["requests"] and phase != "admission"
                if self.publishes and (warm or batch_input["priming"] or phase == "prefill" and not self.model_cut):
                    ctx = stack.enter_context(zmq.Context())
                    receiver = stack.enter_context(ctx.socket(zmq.PULL))
                    receiver.bind(ports.tokenizer_ipc_name)
                if warm:
                    self.warm_cost(batch_input["requests"], receiver)
                ingress, result = self.prepare(batch_input, receiver)

            if self.model_cut:
                boundary = self.observe_model(phase, ingress)
            else:
                boundary = self.observe_host(phase, ingress, result, receiver)

        return boundary
