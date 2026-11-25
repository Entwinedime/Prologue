"""Observe an actual model's sharded embedding without initializing the rest of the decoder."""

import hashlib
import json
from contextlib import ExitStack

import torch
import torch.distributed as distributed

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.call_scopes import NativeCallObservation
from prologue.integrations.frameworks.sglang.observation.communications import NativeCommunications
from prologue.integrations.frameworks.sglang.observation.local.shard_transport import ShardTransport
from prologue.integrations.frameworks.sglang.observation.local.worker_output import (
    activate_device,
    publish_capture,
    publish_facts,
)
from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes
from prologue.integrations.frameworks.sglang.shared.groups import ParallelGroups, WorkerRoles

parallel_state = resolve(symbols().parallel_state)


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
        LoadConfig = resolve(symbols().load_config)
        ModelConfig = resolve(symbols().model_config)
        initialize_dp_attention = resolve(symbols().initialize_dp_attention)
        loader = resolve(symbols().loader)
        ServerArgs = resolve(symbols().server_args)
        set_global_server_args_for_scheduler = resolve(symbols().set_global_server_args_for_scheduler)

        impl = domain["implementation"]
        args = ServerArgs(
            model_path=impl["model"]["_name_or_path"],
            dtype=impl["dtype"].removeprefix("torch."),
            device=platform_domain().device_type,
            tp_size=domain["parallel"]["logical_tp"],
            page_size=impl["page_size"],
            attention_backend=impl["attention_backend"],
            load_format=impl["load_format"],
            skip_tokenizer_init=True,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
        )
        set_global_server_args_for_scheduler(args)
        config = ModelConfig.from_server_args(args)
        if (
            json.loads(config.hf_config.to_json_string(use_diff=False)) != impl["model"]
            or str(config.dtype) != impl["dtype"]
        ):
            raise ValueError("requested and loaded embedding model or dtype differ")
        initialize_dp_attention(args, config)

        with loader.set_default_torch_dtype(config.dtype):
            try:
                with torch.device(platform_domain().device_type):
                    loader._initialize_model(config, LoadConfig(load_format=args.load_format))
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


class EmbeddingRunner:
    def __init__(self, job, output):
        self.job, self.output = job, output

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
                raw = owner.capture.document(env)
                raw.update(markers.timing_evidence(), state_queries=[])
                write_json(output / f"rank-{rank}.json", raw)
                write_json(output / f"rows-{rank}.json", rows.readback(site, result))
                markers.release()
            owner.active = False

    def worker(self, rank):
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
        publish_capture(
            self.output,
            revision,
            0,
            workers=workers,
        )
