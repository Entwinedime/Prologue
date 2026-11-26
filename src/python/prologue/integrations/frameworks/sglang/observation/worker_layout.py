"""Describe the native model, storage and logical rank representatives."""

from __future__ import annotations

from functools import partial, update_wrapper
from inspect import signature
from typing import Any

from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.configuration import ParallelGroups
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout


class LayerCommunication:
    def __init__(self, native) -> None:
        self.native = native

    @classmethod
    def read(cls, module):
        LayerCommunicator = resolve(symbols().layer_communicator)

        native = getattr(module, "layer_communicator", None)
        return cls(native) if isinstance(native, LayerCommunicator) else None

    def rows(self, mode):
        context = self.native._context

        # Native group widths define token ownership; equal observed dimensions alone do not prove it.
        return ["floor_div", ["mul", "model_tokens", context.process_group_sizes[mode]], context.attn_tp_size]

    def observe(self, owner) -> dict:
        modes = self.native.layer_scatter_modes
        for method, mode in (("prepare_mlp", modes.attn_mode), ("postprocess_layer", modes.mlp_mode)):
            original = getattr(self.native, method)
            hook = partial(self.observe_hidden, owner, self.rows(mode), original)
            setattr(self.native, method, update_wrapper(hook, original))
        return {"input": self.rows(modes.layer_input_mode), "output": self.rows(modes.layer_output_mode)}

    @staticmethod
    def observe_hidden(owner, rows, original, hidden_states, *args, **kwargs):
        if owner.active and owner.selected:
            import torch
            get_attn_tp_context = resolve(symbols().get_attn_tp_context)

            # The ordinary layer ABI owns a flat token axis and a fixed hidden width on this shard.
            with owner.tensors.metadata():
                if isinstance(hidden_states,
                    torch.Tensor) and hidden_states.ndim == 2 and not get_attn_tp_context().input_scattered:
                    owner.tensors.declare_shape(hidden_states, [rows, hidden_states.shape[1]])
        return original(hidden_states, *args, **kwargs)

    def document(self) -> dict:
        native = self.native
        context = native._context
        functions = {}

        for name in (
            "_communicate_simple_fn", "_communicate_with_all_reduce_and_layer_norm_fn",
            "_communicate_summable_tensor_pair_fn", "qkv_latent_func",
        ):
            function = getattr(native, name)
            modes = {}
            if isinstance(function, partial):

                # Native communication factories bind scatter modes that change residual ownership.
                bound = signature(function.func).bind_partial(*function.args, **function.keywords)
                modes = {parameter: mode.name for parameter, mode in bound.arguments.items()}
                function = function.func

            functions[name] = None if function is None else {
                "implementation": f"{function.__module__}.{function.__qualname__}", "bound_modes": modes,
            }

        # Compare initialized policies, not rank placement or the communicator's mutable cache.
        return {
            "implementation": f"{type(native).__module__}.{type(native).__qualname__}",
            "modes": {name: mode.name for name, mode in vars(native.layer_scatter_modes).items()},
            "group_sizes": {mode.name: size for mode, size in context.process_group_sizes.items()},
            "sizes": {name: getattr(context, name)
                for name in ("attn_tp_size", "attn_dp_size", "attn_cp_size", "tp_size")},
            "flags": {name: getattr(native, name) for name in (
                "allow_reduce_scatter", "is_last_layer", "force_layernorm_before_dp_gather",
            )},
            "functions": functions,
        }


class ModuleStructure:
    @classmethod
    def read(cls, module: Any, layers: dict[int, int], storages: dict[int, int]) -> Any:
        import torch
        PPMissingLayer = resolve(symbols().pp_missing_layer)

        if isinstance(module, PPMissingLayer):
            return None
        fields = {}
        identities = {
            "rank",
            "tp_rank",
            "pp_rank",
            "ep_rank",
            "moe_ep_rank",
            "moe_tp_rank",
            "attn_tp_rank",
            "gpu_id",
            "device_id",
            "device_index",
            "local_rank",
            "prefix",
            "layer_name",
        }

        for name, value in sorted(vars(module).items()):
            field = name.lstrip("_")
            if field in identities:
                continue

            if type(value) is int and field in {"layer_id", "layer_idx", "start_layer",
                "end_layer"} and value in layers:

                # A local layer reference is distinct from an unrelated literal with the same numeric value.
                fields[name] = ("local_layer", layers[value])
                continue

            if isinstance(value, torch.device):
                value = value.type
            elif isinstance(value, torch.dtype):
                value = str(value)

            if isinstance(value, (str, int, float, bool, type(None))):
                fields[name] = value
            elif field in {"quant_method", "dispatcher"}:
                fields[name] = f"{type(value).__module__}.{type(value).__qualname__}"

        communicator = LayerCommunication.read(module)
        if communicator is not None:
            fields["layer_communicator"] = communicator.document()

        tensors = []
        for name, value in (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)):
            storage = value.untyped_storage()._cdata
            alias = storages.setdefault(storage, len(storages))
            tensors.append(
                (
                    name,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    str(value.dtype),
                    value.device.type,
                    alias,
                )
            )

        children = []
        for name, child in module.named_children():
            key = cls.read(child, layers, storages)
            if key is not None:
                children.append((str(len(children)) if isinstance(module, torch.nn.ModuleList) else name, key))

        # Placement is bound per worker; native implementation, local geometry and flags remain significant.
        return f"{type(module).__module__}.{type(module).__qualname__}", fields, tensors, children


class NativeWorkerLayout:
    """Read the same initialized worker at a serving or a local acquisition boundary."""

    def __init__(self, runner, revision, rank_offset):
        RadixAttention = resolve(symbols().radix_attention)

        self.model = runner.model
        self.rank_offset = rank_offset
        self.groups = ParallelGroups()
        topology = resolve(symbols().get_parallel)()
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
            "framework_revision": revision,
            "model": runner.model_config.hf_config.to_dict(),
            "dtype": str(runner.model_config.dtype),
            "attention_backend": args.attention_backend,
            "page_size": args.page_size,
            "load_format": args.load_format,
            "disaggregation_mode": args.disaggregation_mode,
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
        get_world_group = resolve(symbols().get_world_group)

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
                [
                    (name, group["role"], len(group["members"]), group["members"][0] == identity)
                    for name, group in facts["groups"].items()
                ],
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
