"""Run isolated native scheduler, batch or PD receiver cuts."""

import json
import os
from contextlib import ExitStack, contextmanager, nullcontext
from time import monotonic_ns

import torch
import zmq

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.abi.hooks import HookSpec, HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.call_ports import (
    NativeModelObservation,
)
from prologue.integrations.frameworks.sglang.observation.communications import NativeCommunications
from prologue.integrations.frameworks.sglang.observation.host_timing import NativeHostSamples
from prologue.integrations.frameworks.sglang.observation.model import ModelScopes
from prologue.integrations.frameworks.sglang.observation.serving import FlowScopes
from prologue.integrations.frameworks.sglang.observation.window import (
    LocalProbe,
    NativeProfiles,
    activate_device,
    publish_capture,
    publish_facts,
)
from prologue.integrations.frameworks.sglang.observation.worker_layout import NativeWorkerLayout
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration

BatchTokenIDOutput = resolve(symbols().batch_token_id_output)


sock_recv = resolve(symbols().sock_recv)


GenerationBatchResult = resolve(symbols().generation_batch_result)


DynamicGradMode = resolve(symbols().dynamic_grad_mode)


def install_batch_observer(probe, model, scheduler):
    if probe.model:
        ModelScopes.install_batch_hooks(model)
        if probe.description["worker_count"] > 1:
            model.communications.install()
    else:
        scheduler.install_calls()
    model.request_state.install()


class SchedulerBatchObservation(NativeModelObservation):
    def __init__(self, output, description, *, model):
        super().__init__(device_runtime().markers(detailed_timing=False), "null")
        self.output, self.description, self.model = output, description, model
        self.aliases, self.routing = {}, None
        self.calls = FlowScopes(self)
        self.communications = NativeCommunications(self)

    def install(self):
        install_batch_observer(self, self, self.calls)

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
        with device_runtime().scoped_tasks.profile(self.output, self.parallel["rank"]), runtime.window():
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
        raw = self.document(self.environment, self.calls.queries)
        write_json(self.output / f"rank-{rank}.json", raw)
        runtime.release()
        publish_facts(self.output, rank, raw, self.output / "sdk", "facts.json")


class SchedulerBatchHostTiming:
    def __init__(self, output, description, *, model):
        self.output, self.description, self.model = output, description, model
        self.aliases = {}
        self.calls = NativeHostSamples(self.aliases, description, device_runtime().host_profile)

    def install(self):
        install_batch_observer(self, self.calls, self.calls)

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
        write_json(
            self.output / "host.json", self.calls.document() | {"rank": self.rank, "environment": self.environment}
        )


class SchedulerBatch:
    def __init__(self, scheduler, observer, *, model, observed):
        self.scheduler, self.observer = scheduler, observer
        self.aliases, self.model_cut = observer.aliases, model
        self.record = observer.recording if observed else nullcontext
        self.publishes = scheduler.ipc_channels.send_to_detokenizer.socket is not None
        self.preparation = None
        self.expected = []

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
        if self.step(tokens, receiver) is None:
            raise ValueError("the native scheduler ran out of work while preparing local request progress")

    def step(self, tokens, receiver, *, skip_model=False):
        scheduler = self.scheduler
        batch = scheduler.cur_batch = scheduler.get_next_batch_to_run()
        if batch is not None:
            return self.advance(batch, tokens, receiver, skip_model=skip_model)

        scheduler.on_idle()
        scheduler.last_batch = None
        return None

    def advance(self, batch, tokens, receiver, *, skip_model=False):
        scheduler = self.scheduler
        supplied = torch.tensor([tokens[req.rid] for req in batch.reqs], dtype=torch.int64, device=scheduler.device)
        worker = scheduler.model_worker
        forward = worker.forward_batch_generation

        def supply(batch, **kwargs):
            if skip_model or not self.model_cut:
                return GenerationBatchResult(next_token_ids=supplied)

            # Future model work reads real KV. Let the native forward fill it before publishing the input token.
            output = forward(batch, **kwargs)
            output.next_token_ids.copy_(supplied)
            return output

        with HookTable.scoped(worker, HookSpec("forward_batch_generation", supply, "replace")):
            scheduler.process_batch_result(batch, scheduler.run_batch(batch))
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

    def warm_cost(self, inputs, receiver):
        started = monotonic_ns()
        scheduler = self.scheduler
        allocator, slots = scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
        free = allocator.available_size(), slots.available_size()

        if self.model_cut:

            # Specializations depend on the selected batch, including its membership after native priming.
            ingress, tokens = self.prepare(inputs, receiver)
            batch = self.select(inputs["phase"], ingress)
            warmed = BatchTensors.members(batch)
            self.advance(batch, tokens, receiver)
            expected, batches = None, 1
        else:
            requests, tokens = self.operands(inputs["requests"], output_tokens=2)
            scheduler.process_input_requests(list(requests.values()))
            warmed, expected, batches = None, requests.keys(), 0

        # Finish warmup through native result processing without running unrelated model work.
        finished = set()
        productive = True
        while True:
            published = self.step(tokens, receiver, skip_model=True)
            if published is None:
                if scheduler.is_fully_idle():
                    break
                if not productive:
                    raise ValueError("native warmup could not admit or advance its remaining requests")

                # Filtering a full but finished Decode batch can free slots for the following selection.
                productive = False
                continue

            productive = True
            finished.update(row["id"] for row in published if row["finished"])
            batches += 1

        device_runtime().api.synchronize()
        if self.publishes and expected is not None and finished != expected:
            raise ValueError("native warmup did not complete every local request")
        if not scheduler.flush_cache(empty_cache=False) or (allocator.available_size(), slots.available_size()) != free:
            raise ValueError("native warmup did not restore the original free slots and KV capacity")

        self.preparation = {
            "requests": len(inputs["requests"]),
            "native_batches": batches,
            "free_tokens": free[0],
            "free_slots": free[1],
            "elapsed_ns": monotonic_ns() - started,
        }
        if warmed is not None:
            self.preparation["model_batch"] = warmed

    def prepare_cohort(self, cohort, requests, tokens, receiver):
        self.scheduler.process_input_requests([requests[identity] for identity in cohort["admit"]])
        self.prime_batch(tokens, receiver)
        for _ in range(cohort["decode_steps"]):
            self.prime_batch(tokens, receiver)

    def replay(self, plan, inputs, receiver):
        requests, tokens = self.operands([*plan["prior_requests"], *inputs["requests"]])
        arrivals = {}
        for identity, step in plan["admissions"]:
            arrivals.setdefault(step, []).append(requests[identity])

        # Empty selections also update native scheduler state. Only the demanded cut contributes evidence.
        for step in range(plan["steps"]):
            if incoming := arrivals.get(step):
                self.scheduler.process_input_requests(incoming)
            self.step(tokens, receiver)
        return arrivals.get(plan["steps"], []), tokens

    def prepare(self, inputs, receiver):
        scheduler = self.scheduler
        self.expected = [row["id"] for row in inputs["requests"]]

        if plan := inputs.get("replay"):
            ingress, tokens = self.replay(plan, inputs, receiver)
            if ingress:
                scheduler.process_input_requests(ingress)
            ingress = []
        else:
            requests, tokens = self.operands(inputs["requests"])
            admitted = set()

            for cohort in inputs["priming"]:
                self.prepare_cohort(cohort, requests, tokens, receiver)
                admitted.update(cohort["admit"])

            if inputs["phase"] == "scheduler" and inputs["waiting"]:
                scheduler.process_input_requests([requests[rid] for rid in inputs["waiting"]])
            ingress = [request for identity, request in requests.items() if identity not in admitted]

        return ingress, tokens

    def select(self, phase, ingress):
        scheduler = self.scheduler
        if phase in {"admission", "prefill", "mixed"}:
            scheduler.process_input_requests(ingress)

        if phase == "admission":
            return None

        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if phase == "idle":
            if batch is not None:
                raise ValueError("native worker selected a batch after its idle-pool flush")
        elif phase not in {"selection", "scheduler"} and (
            batch is None or [req.rid for req in batch.reqs] != self.expected or BatchTensors.phase(batch) != phase
        ):
            raise ValueError("native selection differs from the requested phase or ordered members")

        return batch

    def observe_host(self, phase, ingress, result, receiver):
        with self.record():
            if phase == "scheduler":
                queues = self.observer.calls.scheduler_work()["context"]
                batch = self.select(phase, ingress)

                # The serving loop runs the native idle path after a selection without a batch.
                if batch is None:
                    self.scheduler.on_idle()

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
                needs_output = phase in {"prefill", "mixed"} and not self.model_cut
                prepares_requests = bool(batch_input.get("priming") or batch_input.get("replay"))
                if self.publishes and (warm or prepares_requests or needs_output):
                    ctx = stack.enter_context(zmq.Context())
                    receiver = stack.enter_context(ctx.socket(zmq.PULL))
                    receiver.bind(ports.tokenizer_ipc_name)

                if warm:
                    self.warm_cost(batch_input, receiver)
                ingress, tokens = self.prepare(batch_input, receiver)

                if not self.model_cut and phase in {"prefill", "decode", "mixed"}:
                    result = GenerationBatchResult(next_token_ids=torch.tensor(
                        [tokens[identity] for identity in self.expected], dtype=torch.int64, device=scheduler.device,
                    ))

            if self.model_cut:
                return self.observe_model(phase, ingress)
            return self.observe_host(phase, ingress, result, receiver)


class SchedulerRunner(LocalProbe):
    def __init__(self, job, output):
        self.model = job["model"]
        self.configuration = NativeConfiguration(job["configuration"])
        self.batch = job.get("batch")
        self.staged = self.configuration.tp > 1 and job["single_device"]
        self.output = output

    def prepare_pools(self, scheduler, args):
        one_batch = resolve(symbols().one_batch)

        # The control loop starts with empty request and KV pools after one real native forward in each phase.
        runner = scheduler.tp_worker.model_runner
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(1, 1, [[0]])
        tokens, _, batch = one_batch.extend(reqs, runner)
        one_batch.decode(tokens, batch, runner)
        one_batch.synchronize(runner.device)

        if not scheduler.flush_cache(empty_cache=False):
            raise ValueError("native scheduler refused to reset its idle pools after the local warmup")
        one_batch.synchronize(runner.device)

        allocator, slots = scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
        readback = {
            "logical_kv": "empty_after_native_flush", "free_tokens": allocator.available_size(),
            "free_slots": slots.available_size(),
            "workspace": {"prefill_tokens": [len(req.origin_input_ids) for req in batch.reqs], "decode_steps": 1,
                          "device_drained": True, "device_cache_emptied": False},
        }
        settings = {key: getattr(args, key) for key in self.configuration.settings()
                    if key not in {"prefill_chunk_tokens", "microbatch_requests"}}
        settings["prefill_chunk_tokens"] = scheduler.chunked_prefill_size or 0

        if "microbatch_requests" in self.configuration.settings():
            settings["microbatch_requests"] = args.pp_max_micro_batch_size
        return {
            "capacity": BatchTensors.capacity(allocator, slots),
            "configured_settings": settings,
            "initial_state": readback,
        }

    def worker(self, rank, revision, args, ports, host_timing):
        import torch.distributed as distributed
        get_world_group = resolve(symbols().get_world_group)
        parallel_state = resolve(symbols().parallel_state)

        world = self.configuration.tp
        directory = self.output if world == 1 else self.output / "workers" / str(rank)
        (directory / "runtime").mkdir(parents=True, exist_ok=True)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(directory / "runtime")
        device = activate_device(rank, world, self.staged)

        model_cut = self.batch is not None and self.batch["cut"] == "model"
        description = {"mode": "null", "rank_offset": 0, "pp_size": args.pp_size, "ep_size": args.ep_size,
                       "worker_count": world, "instance_directory": str(self.output)}
        probe = SchedulerBatchHostTiming if host_timing else SchedulerBatchObservation
        observer = probe(directory, description, model=model_cut)
        started = monotonic_ns()

        if self.staged:
            from prologue.integrations.frameworks.sglang.observation.staging import ShardTransport

            ShardTransport.world(world, rank, (self.output / "rendezvous").as_uri())
            ShardTransport(self.output, *observer.staging()).install(description)

        observer.install()

        # Upstream applies all hooks for a target once. Install observation and transport together before model loading.
        HookTable.apply()
        scheduler = resolve(symbols().scheduler)(args, ports, gpu_id=device, tp_rank=rank, moe_ep_rank=0,
                              pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None)
        initialized = monotonic_ns()

        try:
            runner = scheduler.tp_worker.model_runner
            layout = NativeWorkerLayout(runner, revision, 0)
            layout.pools(scheduler, args)
            modules = dict(runner.model.named_modules())
            _, experts = ModelScopes.expert_structure(modules)

            if experts:
                raise ValueError("this scheduler cut requires a dense model")
            roles = layout.roles(ModelScopes.selected(modules), experts, ()).document(None)
            environment = layout.environment(scheduler, roles["workers"], device_runtime().markers.environment())
            write_json(directory / "binding.json", self.prepare_pools(scheduler, args))
            observer.bind(scheduler, layout, environment)
            prepared = monotonic_ns()

            observed = roles["observed_roles"]
            if world > 1 and self.batch is not None:
                observed = self.batch.get("peer_ranks", [self.batch["rank"]])
                if "peer_ranks" in self.batch and self.staged:
                    raise ValueError("peer service requires simultaneous observation on distinct physical devices")

            for window in NativeProfiles.windows(self.output, observed, self.staged, 1):
                selected = rank in window.observed
                if world > 1:
                    distributed.barrier(group=get_world_group().cpu_group)

                cut = SchedulerBatch(scheduler, observer, model=model_cut, observed=selected)
                boundary = cut.observe(ports, self.batch, warm_cost=host_timing)
                if selected:
                    observer.export()
                    write_json(directory / "boundary.json", boundary)

                if world > 1:
                    distributed.barrier(group=get_world_group().cpu_group)
        finally:
            scheduler.release_host_resources()
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()

        publish_capture(
            directory,
            revision,
            1,
            rank=rank,
            observed=rank in observed,
            binding='binding.json',
            boundary='boundary.json',
            **({'host': 'host.json'} if host_timing else {'facts': 'facts.json'}),
            initialization_ns=initialized - started,
            preparation_and_readback_ns=prepared - initialized,
            capture_and_export_ns=monotonic_ns() - prepared,
            **({'timing_warmup': cut.preparation} if cut.preparation is not None else {}),
        )

    def run(self, revision, *, host_timing=False):
        one_batch = resolve(symbols().one_batch)
        from torch.multiprocessing import spawn

        args = self.runner_args(one_batch, self.configuration, self.model)
        ports = one_batch.PortArgs.init_new(args)
        world = self.configuration.tp
        inputs = (revision, args, ports, host_timing)

        if world == 1:
            self.worker(0, *inputs)
            return

        # Every rank receives the same native group and ports, including peers that only supply operands.
        spawn(self.worker, args=inputs, nprocs=world, join=True)
        paths = {str(rank): f"workers/{rank}" for rank in range(world)}
        workers = {rank: path for rank, path in paths.items()
                   if json.loads((self.output / path / "capture.json").read_bytes())["observed"]}
        self.publish(
            revision, world,
            workers=workers,
            native_participants=list(range(world)),
        )
