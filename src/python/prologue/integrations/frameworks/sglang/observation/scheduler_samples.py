"""Run isolated native scheduler, batch or PD receiver cuts."""

import json
import os
import subprocess
import sys
from array import array
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic_ns

import torch
import zmq

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookSpec, HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.communications import NativeCommunications
from prologue.integrations.frameworks.sglang.observation.layout import NativeWorkerLayout
from prologue.integrations.frameworks.sglang.observation.model import (
    LocalModel,
    ModelScopes,
    NativeCallObservation,
    NativeModelObservation,
)
from prologue.integrations.frameworks.sglang.observation.observers import NativeHostSamples
from prologue.integrations.frameworks.sglang.observation.serving import FlowScopes
from prologue.integrations.frameworks.sglang.observation.window import (
    LocalProbe,
    NativeProfiles,
    activate_device,
    publish_capture,
    publish_facts,
)
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.configuration import MhaKvLayout, NativeConfiguration
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import DisaggregatedCalls


@dataclass(frozen=True, slots=True)
class ReceiverScheduler:
    server_args: object
    enable_hisparse: bool = False


class ReceiverPools:
    def __init__(self, job):
        DecodePreallocQueue = resolve(symbols().decode_prealloc_queue)
        ChunkCache = resolve(symbols().chunk_cache)

        domain, cap = job["domain"], job["capacity"]
        impl = domain["implementation"]
        page = cap["page_size"]
        size = cap["kv_pages"] * page
        loaded = LocalModel(domain, page, disaggregation_mode="decode",
                            disaggregation_transfer_backend=platform_domain().pd_transfer, disable_radix_cache=True)
        settings, self.model = loaded.args, loaded.config
        if not loaded.matches:
            model = loaded.document
            differences = [key for key in model.keys() | impl["model"].keys()
                           if model.get(key) != impl["model"].get(key)]
            raise ValueError(f"receiver model mismatch: fields {sorted(differences)}, dtype {self.model.dtype}")

        layout = MhaKvLayout.read(domain["kv_layout"])
        dtype = getattr(torch, layout.dtype.removeprefix("torch."))
        width = dtype.itemsize * layout.head_num
        self.kv = resolve(symbols().mha_pool)(size, page, dtype, layout.head_num, layout.key_token_bytes // width,
                                      layout.end_layer - layout.start_layer, platform_domain().device_type, False,
                                      v_head_dim=layout.value_token_bytes // width,
                                      start_layer=layout.start_layer, end_layer=layout.end_layer)
        if MhaKvLayout.from_pool(self.kv) != layout:
            raise ValueError("allocated KV layout differs from the requested receiver")
        self.allocator = resolve(symbols().paged_allocator)(
            size, page, torch.int64, platform_domain().device_type, self.kv, False
        )

        # The batch is already admitted. Preparation only sees the native table's total row count.
        self.slots = resolve(symbols().decode_req_to_token_pool)(
            cap["slot_capacity"], cap["sequence_capacity"], platform_domain().device_type, False, 0
        )
        self.cache = ChunkCache(resolve(symbols().cache_init_params)(True, self.slots, self.allocator, page))

        # Bind the production reservation operation without starting its transport queue or a remote scheduler.
        self.reserve = DecodePreallocQueue.__new__(DecodePreallocQueue)
        self.reserve.scheduler = ReceiverScheduler(settings)
        self.reserve.req_to_token_pool = self.slots
        self.reserve.token_to_kv_pool_allocator = self.allocator
        self.reserve.token_to_kv_pool = self.kv

    def batch(self, members):
        Req = resolve(symbols().req)
        ScheduleBatch = resolve(symbols().schedule_batch)
        SamplingParams = resolve(symbols().sampling_params)
        SpeculativeAlgorithm = resolve(symbols().speculative_algorithm)

        requests, locations = [], []
        for i, member in enumerate(members, start=1):
            ids = array("q", (n % self.model.vocab_size for n in range(member["new_tokens"])))
            request = Req(member["id"], "", ids, SamplingParams(temperature=0, max_new_tokens=2))
            loc = self.reserve._pre_alloc(request)

            # KV delivery precedes this cut. Seed distinct real bytes so the final table can be checked by a consumer.
            self.kv.k_buffer.flatten(1, 2).index_fill_(1, loc, i)
            self.kv.v_buffer.flatten(1, 2).index_fill_(1, loc, -i)
            request.output_ids.append(ids[-1])
            request.init_next_round_input(self.cache)
            request.set_extend_range(len(request.prefix_indices), request.kv_committed_len)
            requests.append(request)
            locations.append(loc.tolist())
        batch = ScheduleBatch.init_new(requests, self.slots, self.allocator, self.cache, self.model,
                                       False, SpeculativeAlgorithm.NONE)
        return batch, locations

    def readable(self, batch, locations):
        flat = [loc for row in locations for loc in row]
        if batch.out_cache_loc.tolist() != flat or min(flat) <= 0 or len(set(flat)) != len(flat):
            return False
        for tag, loc in enumerate(batch.out_cache_loc.split([len(row) for row in locations]), start=1):
            for buffer, value in ((self.kv.k_buffer, tag), (self.kv.v_buffer, -tag)):
                if not (buffer.flatten(1, 2).index_select(1, loc) == value).all().item():
                    return False
        return True


class PrebuiltRunner(LocalProbe):
    def __init__(self, job, output):
        super().__init__(job, output)
        self.rank = job["rank"]

    def observe(self, pools):
        ScheduleBatch = resolve(symbols().schedule_batch)

        batches = [(members, *pools.batch(members)) for members in self.job["batches"]]
        markers = device_runtime().markers(detailed_timing=False)
        owner = NativeCallObservation(markers, "decode")
        owner.active = True
        owner.parallel = {**self.job["domain"]["parallel"], "rank": self.rank}
        owner.aliases = {m["id"]: m["id"] for members, _, _ in batches for m in members}
        cuts = FlowScopes(owner)
        markers.synchronize()
        with device_runtime().scoped_tasks.profile(self.output, self.rank), markers.window():
            for _, batch, _ in batches:
                DisaggregatedCalls(cuts).prepare(ScheduleBatch.prepare_for_prebuilt, batch)

        # The final fence and KV reads belong to qualification, outside the captured asynchronous calls.
        finished = device_runtime().api.Event()
        finished.record()
        finished.synchronize()
        environment = {**self.job["domain"], **markers.environment(), "parallel": owner.parallel}
        raw = owner.document(environment, cuts.queries)

        # The local qualifier consumes these physical writes alongside the independent device readback.
        observed = [scope.pop("page_table") for scope in raw["scopes"] if "page_table" in scope]
        write_json(self.output / f"rank-{self.rank}.json", raw)
        markers.release()
        rows = []
        for (members, batch, locations), table in zip(batches, observed, strict=True):
            if table["slots"] != batch.req_pool_indices.tolist():
                raise ValueError("prepared batch slots differ from the native source rows")
            rows.append({"members": members, "request_slots": table["slots"],
                         "parameters": BatchTensors.parameters(batch, batch.extend_num_tokens),
                         "writes": table["writes"], "verified_kv": pools.readable(batch, locations),
                         "completion": "device_event_synchronized_before_consumer_read"})
        write_json(self.output / "receiver.json", rows)
        return raw

    def run(self, revision):
        ScheduleBatch = resolve(symbols().schedule_batch)

        started = monotonic_ns()
        device_runtime().api.set_device(0)
        pools = ReceiverPools(self.job)
        initialized = monotonic_ns()
        warmup, _ = pools.batch(self.job["batches"][0])
        ScheduleBatch.prepare_for_prebuilt(warmup)
        device_runtime().markers.synchronize()
        pools.allocator.clear()
        pools.slots.clear()
        warmed = monotonic_ns()
        raw = self.observe(pools)

        exporting = monotonic_ns()
        publish_facts(self.output, self.rank, raw, self.output / "sdk", "facts.json")
        self.publish(
            revision, 0,
            measurements='receiver.json',
            facts='facts.json',
            initialization_ns=initialized - started,
            warmup_ns=warmed - initialized,
            capture_ns=exporting - warmed,
            postprocessing_ns=monotonic_ns() - exporting,
        )


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
        device_runtime().api.synchronize()
        if self.publishes and finished != requests.keys():
            raise ValueError("native warmup did not complete every local request")
        if not scheduler.flush_cache(empty_cache=False) or (allocator.available_size(), slots.available_size()) != free:
            raise ValueError("native warmup did not restore the original free slots and KV capacity")
        self.preparation = {
            "requests": len(requests),
            "native_batches": batches,
            "free_tokens": free[0],
            "free_slots": free[1],
        }

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
            result = GenerationBatchResult(
                next_token_ids=torch.tensor(list(tokens.values()), dtype=torch.int64, device=scheduler.device)
            )
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
            batch is None or [req.rid for req in batch.reqs] != list(self.aliases) or BatchTensors.phase(batch) != phase
        ):
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
                return self.observe_model(phase, ingress)
            return self.observe_host(phase, ingress, result, receiver)


class WorkerProbes:
    def __init__(self, revision):
        self.revision = revision

    def run(self, request, output, mode):
        subprocess.run([sys.executable, "-m", "prologue.workers.capture", "capture", "--input", str(request),
                        "--output", str(output), mode], check=True)
        manifest = json.loads((output / "capture.json").read_bytes())
        if manifest["framework_revision"] != self.revision or manifest["engine_starts"]:
            raise ValueError("local probes must execute native worker cuts on the same framework")
        return manifest

    def pair(self, request, output):
        modes = {"structure": "--local-worker", "timing": "--local-host"}

        # Separate processes keep structure instrumentation outside the CPU service measurement.
        starts = sum(self.run(request, output / name, mode)["model_runner_starts"] for name, mode in modes.items())
        publish_capture(output, self.revision, starts, probes={name: name for name in modes})

    def calibration(self, job, output):
        cuts = job.pop("calibration")
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output) as temporary:
            request = Path(temporary) / "job.json"
            write_json(request, job)
            primary = self.run(request, output, "--local-worker")
            paths = {}
            for name, batch in cuts.items():
                paths[name] = f"calibration/{name}"
                write_json(request, job | {"batch": batch})
                measured = self.run(request, output / paths[name], "--local-host-pair")
                primary["model_runner_starts"] += measured["model_runner_starts"]
        write_json(output / "capture.json", primary | {"calibration": paths})


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
