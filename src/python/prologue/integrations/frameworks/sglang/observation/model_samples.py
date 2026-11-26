"""Measure targeted native attention and embedding cuts with physical readback."""

import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from functools import partial
from pathlib import Path
from time import monotonic_ns

import torch
import torch.distributed as distributed

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
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

    def invoke(self, site, module, original, q, k, v, forward_batch, *args, **kwargs):

        if (self.window is not None and self.window.phase != "source") or not forward_batch.forward_mode.is_decode():
            return original(q, k, v, forward_batch, *args, **kwargs)
        lengths = forward_batch.seq_lens_cpu
        context = tuple(int(length) - 1 for length in lengths.tolist()) if lengths is not None else ()
        query_rows = int(q.shape[0])
        key = site, tuple(sorted(context)), query_rows
        self.calls[key] += 1

        # Repeating the geometry measures the unprofiled envelope before recording device tasks.
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
        profile = None

        if self.calls[key] == 2:
            root = Path("cuts") / f"rank-{self.rank}" / str(len(self.rows))
            row["profile"] = str(root)
            profile = device_runtime().scoped_tasks.profile(self.directory / root, self.rank)

        begin, end = (device_runtime().api.Event(enable_timing=True) for _ in range(2))
        device_runtime().api.current_stream().synchronize()
        marker = torch.autograd.profiler.record_function if profile is not None else nullcontext
        with profile if profile is not None else nullcontext():
            with marker("prologue.sample.begin"):
                begin.record()
            with marker("prologue.sample.call"):
                result = original(q, k, v, forward_batch, *args, **kwargs)

            with marker("prologue.sample.end"):
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
    def __init__(self, native, runner):
        self.native, self.runner = native, runner
        self.model = runner.torch_runner
        self.cache = native.TreeCacheNamespace(page_size=self.model.server_args.page_size, device=self.model.device,
                                              token_to_kv_pool_allocator=self.model.token_to_kv_pool_allocator)

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
        batch.prepare_for_extend()

        if batch.input_ids is None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None
        self.forward(batch)
        return batch

    def decode(self, batch):

        # The local dense work query fixes geometry. Feed a declared token without running an unrelated logits head.
        tokens = [req.origin_input_ids[-1] for req in batch.reqs]
        batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=self.model.device)
        batch.prepare_for_decode()
        self.forward(batch)


class KvReadback:
    def __init__(self, pool, module):
        self.pool = pool
        self.layer = module.layer_id - pool.start_layer
        self.operands = None

    def retain(self, module, args):
        self.operands = args[1:3]

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
    def measure(self, lengths, observer, kv):
        started = monotonic_ns()
        batch = self.batch.extend(lengths)
        prefill = kv.read(batch, None, True)
        prepared = monotonic_ns()
        self.batch.decode(batch)
        decoded = monotonic_ns()
        read = kv.read(batch, prefill["locations"], False)
        call = observer.rows[-1]
        call["kv"] = {"prefill": prefill, "decode": read}

        return {"context": lengths, "prepare_and_read_ns": prepared - started,
                "decode_ns": decoded - prepared, "readback_ns": monotonic_ns() - decoded}

    def run(self, revision):
        started = monotonic_ns()
        one_batch = resolve(symbols().one_batch)

        config = NativeConfiguration(self.job["runner"]["configuration"])
        args = self.runner_args(one_batch, config, self.job["runner"]["model"])
        one_batch.initialize_moe_config(args)
        one_batch.initialize_fp8_gemm_config(args)
        one_batch.initialize_fp4_gemm_config(args)
        runner, _ = one_batch.load_model(args, one_batch.PortArgs.init_new(args), gpu_id=0, tp_rank=0)
        initialized = monotonic_ns()

        try:
            actual = runner.torch_runner
            self.batch = NativeAttentionBatch(one_batch, runner)
            site = self.job["site"]
            module = actual.model.get_submodule(site)

            with module.register_forward_hook(self.batch.stop_after_call):
                for lengths in self.job["contexts"]:
                    self.batch.decode(self.batch.extend(lengths))
                warmed = monotonic_ns()
                observer = DecodeAttentionObservation(
                    None, {"output": self.output, "rank_offset": 0, "attention_sites": {"0": [site]}}
                )
                observer.attach(actual)
                kv = KvReadback(actual.token_to_kv_pool, module)
                timings = []

                with module.register_forward_pre_hook(kv.retain):
                    for lengths in self.job["contexts"]:
                        for _ in range(2):
                            timings.append(self.measure(lengths, observer, kv))

            record = observer.document()
            record.update(
                model=actual.model_config.hf_config.to_dict(), rank=observer.rank, load_format=args.load_format
            )
            runner.clear()
        finally:
            one_batch.destroy_model_parallel()
            one_batch.destroy_distributed_environment()

        # SDK postprocessing starts after the measured worker has released its process group.
        exporting = monotonic_ns()
        device_runtime().scoped_tasks.export_calls(self.output, observer.rank, record["calls"])
        write_json(self.output / "attention.json", record)
        self.publish(
            revision, 1,
            measurements='attention.json',
            initialization_ns=initialized - started,
            warmup_ns=warmed - initialized,
            points=timings,
            postprocessing_ns=monotonic_ns() - exporting,
        )


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
        from torch.multiprocessing import spawn

        world = self.job["domain"]["parallel"]["logical_tp"]
        spawn(self.worker, nprocs=world, join=True)
        workers = {}
        for rank in self.job["ranks"]:
            raw = json.loads((self.output / f"rank-{rank}.json").read_bytes())
            name = f"facts-{rank}.json"
            publish_facts(self.output, rank, raw, self.output / f"sdk-{rank}", name)
            workers[str(rank)] = {"facts": name, "readback": f"rows-{rank}.json"}

        self.publish(revision, 0, workers=workers)
