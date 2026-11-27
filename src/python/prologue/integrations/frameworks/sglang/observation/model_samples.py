"""Measure targeted native attention and embedding cuts with physical readback."""

import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from time import monotonic_ns

import torch
import torch.distributed as distributed
from torch.multiprocessing import spawn

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookSpec, HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.call_ports import LocalModel, NativeCallObservation
from prologue.integrations.frameworks.sglang.observation.communications import NativeCommunications
from prologue.integrations.frameworks.sglang.observation.model import ModelScopes
from prologue.integrations.frameworks.sglang.observation.staging import ShardTransport
from prologue.integrations.frameworks.sglang.observation.window import (
    LocalProbe,
    activate_device,
    publish_facts,
)
from prologue.integrations.frameworks.sglang.shared.configuration import (
    NativeConfiguration,
    ParallelGroups,
    WorkerRoles,
)
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout


class DecodeAttentionObservation:
    def __init__(self, window, description):
        self.window = window
        self.directory = Path(description["output"])
        self.offset = description["rank_offset"]
        self.sites = description["attention_sites"]
        self.calls = defaultdict(int)
        self.rows = []
        self.batch_profile = None

    def load_model(self, original, runner):
        result = original(runner)
        self.attach(runner)
        return result

    def attach(self, runner):
        self.rank = self.offset + runner.pp_rank * runner.tp_size + runner.tp_rank
        self.backend = runner.server_args.attention_backend
        modules = dict(runner.model.named_modules())
        selected = ModelScopes.selected(modules)
        sites = self.sites.get(str(self.rank), ())
        for site in sites:
            if selected.get(site) != "attention":
                raise ValueError(f"requested site is not an observed native attention module: {site}")
            module = modules[site]
            module.forward = partial(self.invoke, site, module, module.forward)

    def profile_path(self, name):
        return Path("cuts") / f"rank-{self.rank}" / str(name)

    @contextmanager
    def profile_batch(self):
        self.batch_profile = self.profile_path("batch")
        try:
            with device_runtime().scoped_tasks.profile(self.directory / self.batch_profile, self.rank):
                yield
        finally:
            self.batch_profile = None

    def invoke(self, site, module, original, q, k, v, forward_batch, *args, **kwargs):

        if (self.window is not None and self.window.phase != "source") or not forward_batch.forward_mode.is_decode():
            return original(q, k, v, forward_batch, *args, **kwargs)
        lengths = forward_batch.seq_lens_cpu
        context = tuple(int(length) - 1 for length in lengths.tolist()) if lengths is not None else ()
        query_rows = int(q.shape[0])
        key = site, tuple(sorted(context)), query_rows
        self.calls[key] += 1

        pool = resolve(symbols().get_attn_backend)().token_to_kv_pool
        row = {
            "site": site,
            "implementation": f"{type(module).__module__}.{type(module).__qualname__}",
            "context": list(context),
            "query_rows": query_rows,
            "kv_layout": MhaKvLayout.from_pool(pool).document(),
            "backend": self.backend,
            "page_size": pool.page_size,
            "operands": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in (("argument.q", q), ("argument.k", k), ("argument.v", v))
                if tensor is not None
            },
            "full_context": not module.is_cross_attention and module.sliding_window_size < 0,
            "heads": {
                "query": module.tp_q_head_num,
                "key": module.tp_k_head_num,
                "value": module.tp_v_head_num,
                "key_dim": module.qk_head_dim,
                "value_dim": module.v_head_dim,
            },
        }

        profile = nullcontext()
        marker = nullcontext
        scope = f"prologue.sample/{len(self.rows)}"

        # Measure the unprofiled envelope before recording device tasks for the same geometry.
        if self.calls[key] == 2:
            root = self.batch_profile or self.profile_path(len(self.rows))
            row.update(profile=str(root), scope=scope)
            marker = torch.autograd.profiler.record_function
            if self.batch_profile is None:
                profile = device_runtime().scoped_tasks.profile(self.directory / root, self.rank)

        begin, end = (device_runtime().api.Event(enable_timing=True) for _ in range(2))
        device_runtime().api.current_stream().synchronize()
        with profile:
            with marker(f"{scope}/begin"):
                begin.record()
            with marker(f"{scope}/call"):
                result = original(q, k, v, forward_batch, *args, **kwargs)

            with marker(f"{scope}/end"):
                end.record()
            end.synchronize()

        row["envelope_ns"] = round(begin.elapsed_time(end) * 1_000_000)
        self.rows.append(row)
        return result

    def document(self):
        return {"runtime": device_runtime().markers.environment(), "calls": self.rows}

    def install(self):
        target = symbols().model_runner_load_model
        HookTable.around(target, self.load_model)


class AttentionReturned(Exception):
    """The requested native layer has committed its output; later model layers are outside the cut."""


class NativeAttentionBatch:
    def __init__(self, native, runner, module):
        self.native, self.runner = native, runner
        self.model = runner.torch_runner
        self.module = module
        self.query = (module.tp_q_head_num, module.qk_head_dim)
        self.key = (module.tp_k_head_num, module.qk_head_dim)
        self.value = (module.tp_v_head_num, module.v_head_dim)
        self.generator = torch.Generator(device="cpu").manual_seed(0)
        self.kv = KvReadback(self.model.token_to_kv_pool, module)
        self.allocator = self.model.token_to_kv_pool_allocator
        self.cache = native.TreeCacheNamespace(page_size=self.model.server_args.page_size, device=self.model.device,
                                              token_to_kv_pool_allocator=self.allocator)

    def operand(self, rows, geometry):
        heads, width = geometry
        return torch.rand((rows, heads * width), generator=self.generator, device="cpu",
                          dtype=self.model.model_config.dtype).to(self.model.device)

    def attention(self, input_ids, positions, forward_batch, **kwargs):

        # Keep the native runner's batch metadata and KV writes while supplying only the target call's operands.
        rows = input_ids.numel()
        key, value = self.kv.operands = self.operand(rows, self.key), self.operand(rows, self.value)
        return self.module(self.operand(rows, self.query), key, value, forward_batch)

    @staticmethod
    def stop_after_call(module, args, result):
        raise AttentionReturned

    def forward(self, batch):

        self.native._maybe_prepare_mlp_sync_batch(batch, self.model)
        forward = self.native.ForwardBatch.init_new(batch, self.model)
        try:
            with torch.no_grad():
                self.model.forward(forward)
        except AttentionReturned:
            self.runner.synchronize()
            return
        raise ValueError("the native forward did not reach the requested attention site")

    def allocate_prefix(self, prefix, prefix_cpu, sequences, sequences_cpu, last, count):
        lengths, page = sequences_cpu.tolist(), self.allocator.page_size
        if any(prefix_cpu.tolist()) or sum(lengths) != count:
            raise ValueError("local attention preparation requires fresh native request pages")

        # A fresh prefix starts at page boundaries, so the native allocator can reserve whole pages directly.
        sizes = [((length + page - 1) // page) * page for length in lengths]
        pages = self.allocator.alloc(sum(sizes))
        if pages is None:
            return None

        return torch.cat([slots[:length] for slots, length in zip(pages.split(sizes), lengths, strict=True)])

    def extend(self, lengths):
        self.runner.clear()
        vocab = self.model.model_config.vocab_size
        ids = [[(index + pos) % vocab for pos in range(n)] for index, n in enumerate(lengths)]
        requests = self.native.prepare_synthetic_inputs_for_latency_test(len(ids), max(lengths), ids)
        batch = self.native.ScheduleBatch.init_new(
            reqs=requests,
            req_to_token_pool=self.model.req_to_token_pool,
            token_to_kv_pool_allocator=self.model.token_to_kv_pool_allocator,
            tree_cache=self.cache,
            model_config=self.model.model_config,
            enable_overlap=False,
            spec_algorithm=self.native.SpeculativeAlgorithm.NONE,
        )
        if self.allocator.page_size > 1:
            allocate = HookSpec("alloc_extend", self.allocate_prefix, "replace")
            with HookTable.scoped(self.allocator, allocate):
                batch.prepare_for_extend()
        else:
            batch.prepare_for_extend()

        # Decode needs the prefix in its assigned slots. The native pool can write it without a Prefill forward.
        rows = batch.out_cache_loc.numel()
        key, value = self.kv.operands = self.operand(rows, self.key), self.operand(rows, self.value)
        self.kv.pool.set_kv_buffer(self.module, batch.out_cache_loc, key.view(rows, *self.key),
                                   value.view(rows, *self.value))
        self.runner.synchronize()
        return batch

    def decode(self, batch):

        # The local dense work query fixes geometry. Feed a declared token without running an unrelated logits head.
        tokens = [req.origin_input_ids[-1] for req in batch.reqs]
        batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=self.model.device)
        batch.prepare_for_decode()
        self.forward(batch)

    def measure(self, lengths, observer):
        started = monotonic_ns()
        batch = self.extend(lengths)
        prefill = self.kv.read(batch, None, True)
        prepared = monotonic_ns()
        self.decode(batch)
        decoded = monotonic_ns()
        read = self.kv.read(batch, prefill["locations"], False)
        observer.rows[-1]["kv"] = {"prefill": prefill, "decode": read}

        return {"context": lengths, "prepare_and_read_ns": prepared - started,
                "decode_ns": decoded - prepared, "readback_ns": monotonic_ns() - decoded}


class KvReadback:
    def __init__(self, pool, module):
        self.pool = pool
        self.layer = module.layer_id - pool.start_layer
        self.operands = None

    def read(self, batch, locations, extend):

        slots = batch.req_pool_indices_cpu.tolist()
        lengths = batch.seq_lens_cpu.tolist()
        table = [
            batch.req_to_token_pool.req_to_token[slot, :length].tolist()
            for slot, length in zip(slots, lengths, strict=True)
        ]
        written = batch.out_cache_loc.tolist()
        flat = [loc for row in table for loc in row]
        expected = flat if extend else [row[-1] for row in table]
        correct = (written == expected and len(set(flat)) == len(flat) and min(flat) > 0
                   and lengths == [request.kv_committed_len for request in batch.reqs])
        matching_prefix = extend or [row[:-1] for row in table] == locations

        # Read just the requested slots after the device interval; preserve both keys and values for the next read.
        pool = self.pool
        contents = []
        for buffer, operand in zip((pool.k_buffer, pool.v_buffer), self.operands, strict=True):
            tokens = buffer[self.layer].flatten(0, 1)
            now = tokens[flat].cpu()
            expected_data = operand.reshape(-1, *tokens.shape[1:]).cpu()

            if extend:
                correct = correct and torch.equal(now, expected_data)
            else:
                pieces = now.split(lengths)
                correct = correct and torch.equal(torch.stack([piece[-1] for piece in pieces]), expected_data)
                matching_prefix = matching_prefix and torch.equal(
                    torch.cat([p[:-1] for p in pieces]), self.prefix[len(contents)]
                )

            contents.append(now)

        self.operands = None
        if extend:
            self.prefix = contents
        return {"slots": slots, "locations": table, "written": written, "committed": lengths,
                "matching_operands": correct, "preserved_prefix": matching_prefix}


class AttentionRunner(LocalProbe):
    def observe(self, one_batch, runner, rank, initialized):
        actual = runner.torch_runner
        site, contexts = self.job["site"], self.job["contexts"]
        module = actual.model.get_submodule(site)
        cut = NativeAttentionBatch(one_batch, runner, module)
        forward = HookSpec("forward", cut.attention, "replace")
        warmup = []

        with HookTable.scoped(actual.model, forward), module.register_forward_hook(cut.stop_after_call):
            for lengths in contexts:
                start = monotonic_ns()
                batch = cut.extend(lengths)
                extended = monotonic_ns()
                cut.decode(batch)
                warmup.append({"context": lengths, "prefill_ns": extended - start,
                               "decode_ns": monotonic_ns() - extended})

            warmed = monotonic_ns()
            observer = DecodeAttentionObservation(
                None, {"output": self.output, "rank_offset": 0, "attention_sites": {str(rank): [site]}}
            )
            observer.attach(actual)
            timings = [cut.measure(lengths, observer) for lengths in contexts]

            # Each cut retains its own markers inside the shared device profile.
            with observer.profile_batch():
                timings.extend(cut.measure(lengths, observer) for lengths in contexts)

        record = observer.document()
        record.update(model=actual.model_config.hf_config.to_dict(), rank=observer.rank,
                      load_format=actual.server_args.load_format)
        runner.clear()
        return record, {"warmup_ns": warmed - initialized, "warmup": warmup, "points": timings}

    def worker(self, rank, revision, args, ports, started):
        one_batch = resolve(symbols().one_batch)
        world, selected = args.tp_size, self.job["rank"]
        staged = world > 1
        device = activate_device(rank, world, staged)
        if staged:
            ShardTransport.world(world, rank, (self.output / "rendezvous").as_uri())
            transport = ShardTransport(self.output, nullcontext)
            transport.install({"ep_size": 1, "instance_directory": self.output, "worker_count": world})
            HookTable.apply()

        one_batch.initialize_moe_config(args)
        one_batch.initialize_fp8_gemm_config(args)
        one_batch.initialize_fp4_gemm_config(args)

        try:
            runner, _ = one_batch.load_model(args, ports, gpu_id=device, tp_rank=rank)
            initialized = monotonic_ns()
            if rank == selected:
                if staged:
                    transport.local_only = True
                record, timing = self.observe(one_batch, runner, rank, initialized)

            # Peers supply native initialization but stay off the device during the measured local cut.
            if staged:
                distributed.barrier()
        finally:
            one_batch.destroy_model_parallel()
            one_batch.destroy_distributed_environment()

        if rank == selected:
            self.export(revision, world, record, timing | {"initialization_ns": initialized - started})

    def export(self, revision, world, record, timing):
        exporting = monotonic_ns()
        try:
            device_runtime().scoped_tasks.export_calls(self.output, record["rank"], record["calls"])
        finally:
            write_json(self.output / "attention.json", record)

        self.publish(revision, world, measurements="attention.json", **timing,
                     postprocessing_ns=monotonic_ns() - exporting)

    def run(self, revision):
        started = monotonic_ns()
        one_batch = resolve(symbols().one_batch)
        runner = self.job["runner"]
        configuration = NativeConfiguration(runner["configuration"])
        args = self.runner_args(one_batch, configuration, runner["model"])
        worker_args = revision, args, one_batch.PortArgs.init_new(args), started

        # The file rendezvous needs its parent before any worker starts.
        self.output.mkdir(parents=True, exist_ok=True)
        if configuration.tp == 1:
            self.worker(0, *worker_args)
        else:
            spawn(self.worker, args=worker_args, nprocs=configuration.tp, join=True)


class EmbeddingReady(Exception):
    """Return the shard as soon as the upstream model constructor has finished creating it."""

    @staticmethod
    def capture(original, module, *args, **kwargs):
        VocabParallelEmbedding = resolve(symbols().vocab_parallel_embedding)

        original(module, *args, **kwargs)
        if type(module) is VocabParallelEmbedding:
            raise EmbeddingReady(kwargs["prefix"], module)

    @classmethod
    def load(cls, domain, rank):
        loader = resolve(symbols().loader)

        loaded = LocalModel(domain, domain["implementation"]["page_size"])
        args, config = loaded.args, loaded.config
        if not loaded.matches:
            raise ValueError("requested and loaded embedding model or dtype differ")

        resolve(symbols().initialize_dp_attention)(args, config)

        with loader.set_default_torch_dtype(config.dtype):
            try:
                with torch.device(platform_domain().device_type):
                    loader._initialize_model(config, resolve(symbols().load_config)(load_format=args.load_format))
            except EmbeddingReady as ready:
                site, shard = ready.args
            else:
                raise ValueError("model initialization produced no native vocabulary embedding")

            if site not in domain["worker_roles"][str(rank)]["modules"] or shard.tp_size != args.tp_size:
                raise ValueError("embedding shard does not match the declared native worker site and group")
            loader.initialize_dummy_weights(shard)
            shard.quant_method.process_weights_after_loading(shard)

        return site, shard.eval()


class ShardRows:
    def __init__(self, shard, count, rank, cpu_group):
        indices = shard.shard_indices
        self.ranges = [None] * shard.tp_size
        distributed.all_gather_object(
            self.ranges, [indices.org_vocab_start_index, indices.org_vocab_end_index], group=cpu_group
        )
        bounds = [0, *(end for _, end in self.ranges)]

        if (bounds[-1] != shard.org_vocab_size or any(end <= start for start, end in self.ranges)
                or [start for start, _ in self.ranges] != bounds[:-1]):
            raise ValueError("native shards do not partition the model vocabulary")

        ids = []
        for position in range(count):
            start, end = self.ranges[position % shard.tp_size]
            ids.append(start + position % (end - start))

        self.ids = torch.tensor(ids, device=platform_domain().device_type, dtype=torch.int64)

        # Obtain expected values from the owning weight rows, independently of the embedding forward and its transport.
        start, end = self.ranges[rank]
        local = torch.zeros(count, shard.weight.shape[1], dtype=torch.float32)
        positions = [i for i, token in enumerate(ids) if start <= token < end]
        if positions:
            rows = torch.tensor(
                [ids[i] - start for i in positions], device=platform_domain().device_type, dtype=torch.int64
            )
            local[positions] = shard.weight.index_select(0, rows).cpu().float()

        contributions = [torch.empty_like(local) for _ in self.ranges]
        distributed.all_gather(contributions, local, group=cpu_group)
        self.expected = torch.stack(contributions).sum(0)

    def readback(self, site, result):
        actual = result.cpu().contiguous()
        if not torch.equal(actual.float(), self.expected):
            raise ValueError("native embedding output differs from the owning shards' weight rows")
        return {"site": site, "shape": list(actual.shape), "dtype": str(actual.dtype), "ranges": self.ranges,
                "token_ids": self.ids.tolist(), "matched_owning_weights": True,
                "output_sha256": hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()}


class EmbeddingRunner(LocalProbe):
    @torch.no_grad()
    def observe(self, rank):
        job, output = self.job, self.output
        domain = job["domain"]
        role = domain["worker_roles"][str(rank)]
        markers = device_runtime().markers(detailed_timing=False)
        owner = NativeCallObservation(markers, "null")
        owner.description = {"rank_offset": 0, "pp_size": 1}
        owner.parallel, owner.worker_roles = role["parallel"], domain["worker_roles"]
        owner.groups = ParallelGroups()

        if owner.groups.document(0) != role["groups"]:
            raise ValueError("native embedding group differs from the declared role")
        owner.tensors.namespace = f"rank/{rank}/"
        owner.capture.scopes[0]["work"] = {"phase": job["phase"], "rank": rank}
        communication = NativeCommunications(owner)
        transport = None

        if job["single_device"]:
            transport = ShardTransport(output, owner.tensors.auxiliary, communication.local_exchange)
            transport.install_collectives(1)
        communication.cuts.install()
        HookTable.around(symbols().vocab_parallel_embedding_init, EmbeddingReady.capture)
        HookTable.apply()
        site, shard = EmbeddingReady.load(domain, rank)
        ModelScopes(owner).attach(site, "embedding", shard)
        cpu_group = owner.groups.groups["tp_cpu"]
        rows = ShardRows(shard, job["tokens"], rank, cpu_group)

        def lookup():
            return transport.compute(shard, rows.ids) if transport else shard(rows.ids)

        lookup()
        markers.synchronize()
        passes = [[peer] for peer in job["ranks"]] if transport else [job["ranks"]]
        for observed in passes:
            distributed.barrier(group=cpu_group)
            owner.active = rank in observed
            with ExitStack() as stack:
                if owner.active:
                    stack.enter_context(device_runtime().scoped_tasks.profile(output, rank))
                    stack.enter_context(markers.window())
                    stack.enter_context(owner.tensors)

                result = lookup()

            if owner.active:
                markers.synchronize()
                communication.bind_returned_operands()
                env = WorkerRoles.bind_worker(domain, role) | markers.environment()
                raw = owner.document(env, [])
                write_json(output / f"rank-{rank}.json", raw)
                write_json(output / f"rows-{rank}.json", rows.readback(site, result))
                markers.release()

            owner.active = False

    def worker(self, rank):
        parallel_state = resolve(symbols().parallel_state)
        world = self.job["domain"]["parallel"]["logical_tp"]
        staged = self.job["single_device"]
        device = activate_device(rank, world, staged)
        rendezvous = (self.output / "rendezvous").as_uri()

        if staged:
            ShardTransport.world(world, rank, rendezvous)
        else:
            parallel_state.init_distributed_environment(
                world, rank, rendezvous, local_rank=device, backend=platform_domain().collective_backend
            )

        try:
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
            self.observe(rank)
        finally:
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()

    def run(self, revision):
        world = self.job["domain"]["parallel"]["logical_tp"]
        spawn(self.worker, nprocs=world, join=True)
        workers = {}
        for rank in self.job["ranks"]:
            raw = json.loads((self.output / f"rank-{rank}.json").read_bytes())
            name = f"facts-{rank}.json"
            publish_facts(self.output, rank, raw, self.output / f"sdk-{rank}", name)
            workers[str(rank)] = {"facts": name, "readback": f"rows-{rank}.json"}

        self.publish(revision, 0, workers=workers)
