"""Choose representatives before observation and retain actual worker/group bindings."""

from __future__ import annotations

from typing import Any

from prologue.integrations.frameworks.sglang.observation.module_layout import ModuleStructure
from prologue.integrations.frameworks.sglang.groups import ParallelGroups
from prologue.integrations.frameworks.sglang.kv_layout import MhaKvLayout


class NativeWorkerLayout:
    """Read the same initialized worker at a serving or a local acquisition boundary."""

    def __init__(self, runner, revision, rank_offset):
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.runtime_context import get_parallel

        self.model = runner.model
        self.rank_offset = rank_offset
        self.groups = ParallelGroups()
        topology = get_parallel()
        layers = [m.layer_id for m in self.model.modules() if isinstance(m, RadixAttention)]
        self.parallel = {
            "rank": rank_offset + runner.pp_rank * runner.tp_size + runner.tp_rank,
            "tp_rank": runner.tp_rank, "pp_rank": runner.pp_rank,
            "logical_tp": runner.tp_size, "logical_pp": runner.pp_size,
            "logical_ep": topology.moe_ep_size, "ep_rank": topology.moe_ep_rank,
            "layer_range": [min(layers), max(layers) + 1],
        }
        args = runner.server_args
        self.implementation = {
            "framework_revision": revision, "model": runner.model_config.hf_config.to_dict(),
            "dtype": str(runner.model_config.dtype), "attention_backend": args.attention_backend,
            "page_size": args.page_size, "load_format": args.load_format, "disaggregation_mode": args.disaggregation_mode,
        }

    def pools(self, pools, server_args):
        allocator = pools.token_to_kv_pool_allocator
        allocator_type, cache_type = type(allocator), type(pools.tree_cache)
        self.implementation.update(
            allocator=f"{allocator_type.__module__}.{allocator_type.__qualname__}",
            cache={"type": f"{cache_type.__module__}.{cache_type.__qualname__}",
                   "disabled": pools.disable_radix_cache, "eviction_policy": server_args.radix_eviction_policy.lower()},
        )
        self.kv_layout = MhaKvLayout.from_pool(allocator.get_kvcache())

    def roles(self, sites, expert_layers, conditions):
        facts = {
            "parallel": self.parallel, "groups": self.groups.document(self.rank_offset),
            "kv_layout": self.kv_layout.document(), "expert_layers": expert_layers,
            "modules": [path for path, _ in self.model.named_modules() if path in sites],
        }
        return WorkerRolePlan(self.model, facts, conditions)

    def environment(self, scheduler, roles, runtime):
        self.implementation["scheduling"] = {
            "chunked_prefill": scheduler.chunked_prefill_size is not None, "mixed_batches": scheduler.is_mixed_chunk,
        }
        role = roles[str(self.parallel["rank"])]
        return {
            "framework": "sglang", **runtime, "parallel": self.parallel, "implementation": self.implementation,
            "kv_layout": self.kv_layout.document(), "worker_roles": roles,
            **{key: role[key] for key in ("expert_layers", "kv_peers") if key in role},
        }


class WorkerRolePlan:
    def __init__(self, model: Any, facts: dict[str, Any], observation_conditions) -> None:
        import torch.distributed as distributed
        from sglang.srt.distributed import get_world_group

        parallel = facts["parallel"]
        identity = parallel["rank"]
        layout = dict(facts["kv_layout"])
        first, last = layout["start_layer"], layout["end_layer"]
        layers = {layer: index for index, layer in enumerate(range(first, last + 1))}
        layout.update(start_layer=0, end_layer=last - first)
        facts = {
            **facts,
            "observation_conditions": observation_conditions,
            "key": (
                parallel["tp_rank"] == 0,
                parallel["pp_rank"] == 0,
                parallel["pp_rank"] + 1 == parallel["logical_pp"],
                [(name, group["role"], len(group["members"]), group["members"][0] == identity) for name, group in facts["groups"].items()],
                layout,
                ModuleStructure.read(model, layers, {}),
            ),
        }
        group = get_world_group().cpu_group
        self.workers = [None] * distributed.get_world_size(group)

        # PP forwards control requests after local handling, so whole-world exchange belongs to initialization.
        distributed.all_gather_object(self.workers, facts, group=group)

    @staticmethod
    def observed_ranks(workers, bindings):
        selected = {worker["representative"] for worker in bindings.values()}
        for representative in sorted(selected):
            group = {worker["parallel"]["rank"]: set(worker["observation_conditions"]) for worker in workers
                     if bindings[str(worker["parallel"]["rank"])]["representative"] == representative}
            missing = set().union(*group.values()) - group[representative]
            while missing:
                rank = min(group, key=lambda rank: (-len(group[rank] & missing), rank))
                selected.add(rank)
                missing.difference_update(group[rank])
        return sorted(selected)

    def document(self, peers) -> dict[str, Any]:
        representatives = []
        bindings = {}
        for initialized in self.workers:
            worker = dict(initialized)
            if peers is not None:
                rank = worker["parallel"]["rank"]
                worker["kv_peers"] = peers.layouts(rank)
                worker["key"] = (worker["key"], peers.signature(rank))
            source = next((previous for previous in representatives if previous["key"] == worker["key"]), None)
            if source is None:
                source = worker
                representatives.append(worker)
            bindings[str(worker["parallel"]["rank"])] = {
                "representative": source["parallel"]["rank"],
                **{name: value for name, value in worker.items() if name not in {"key", "observation_conditions"}},
            }

        return {
            "basis": "native_module_layout_group_roles_and_source_conditions",
            "participants": len(self.workers),
            "representatives": [worker["parallel"]["rank"] for worker in representatives],
            "observed_roles": self.observed_ranks(self.workers, bindings),
            "workers": bindings,
        }
