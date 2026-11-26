"""Observe routed model work on the same native expert placement as serving."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial, update_wrapper
from typing import Any

import torch

from prologue.common.workload import ExpertRoutes, RoutedContent, WorkloadContent
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.worker_layout import ModuleStructure
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RoutedCacheKeys
from prologue.integrations.frameworks.sglang.shared.expert_layout import ExpertBatch


class ExpertScopes:
    def __init__(self, observation, modules) -> None:
        import os

        MaybeTboDeepEPDispatcher = resolve(symbols().maybe_tbo_deep_ep_dispatcher)
        DeepEPDispatcher = resolve(symbols().deep_ep_dispatcher)
        StandardDispatcher = resolve(symbols().standard_dispatcher)

        self.observation = observation
        self.sites = []
        self.implementations = {}
        geometry = observation.routing.geometry

        if geometry.size > 1 and os.environ.get("DEEP_USE_ALLTOALL_MODE") != "1":
            raise ValueError("expert geometry requires the native DeepEP normal AllToAll layout")
        dispatcher = StandardDispatcher if geometry.size == 1 else MaybeTboDeepEPDispatcher
        observation.implementation["expert_layout"] = geometry.layout

        for path, module in modules.items():
            if not isinstance(module.dispatcher, dispatcher):
                raise ValueError("native expert dispatcher differs from its declared boundary layout")
            if geometry.size > 1:

                # Native FusedMoE owns a wrapper even without overlap; communication hooks observe its single inner
                # dispatcher.
                inner, = module.dispatcher._inners
                if not isinstance(inner, DeepEPDispatcher):
                    raise ValueError("native expert exchange differs from the declared DeepEP layout")

            self.sites.append(path)

            # Equal initialized modules may share local bodies; their concrete layer/weight bindings remain on each
            # call.
            self.implementations[path] = {
                "layout": geometry.layout, "module": ModuleStructure.read(module, {module.layer_id: 0}, {}),
            }
            original = module.forward
            module.forward = update_wrapper(partial(self.forward, path, module, original), original)
            original = module.run_moe_core
            module.run_moe_core = update_wrapper(partial(self.compute, original), original)

    def forward(self, path, module, original, hidden_states, topk_output):
        owner = self.observation
        if not owner.active or not owner.selected:
            return original(hidden_states, topk_output)

        enclosing = owner.capture.scopes[owner.capture.current.get()]
        parameters, relations = owner.routing.geometry.scope(enclosing, module.layer_id)
        parameters["expert_input_tokens"] = parameters["model_tokens"] // module.moe_ep_size
        relations["expert_input_tokens"] = ["floor_div", "model_tokens", module.moe_ep_size]

        with owner.capture.call(
            "experts", path, f"{type(module).__module__}.{type(module).__qualname__}",
            parameters=parameters, arguments=relations, constants=enclosing["constants"], device=True,
            bindings={
                "weights": [name for name, _ in module.named_parameters(recurse=False)],
                "local_implementation": self.implementations[path],
            },
        ) as call:
            self.port(call, "hidden_states", "input", hidden_states, "expert_input_tokens")
            self.port(call, "topk_ids", "input", topk_output.topk_ids, "expert_input_tokens")
            self.port(call, "topk_weights", "input", topk_output.topk_weights, "expert_input_tokens")
            result = original(hidden_states, topk_output)
            self.port(call, "hidden_states", "output", result, "expert_input_tokens")

            return result

    def compute(self, original, dispatch_output):
        owner = self.observation
        if not owner.active or not owner.selected:
            return original(dispatch_output)

        rows = owner.routing.geometry.compute_rows
        with owner.call("expert_compute", original, device=True) as call:
            self.port(call, "hidden_states", "input", dispatch_output.hidden_states, rows)
            if dispatch_output.hidden_states_scale is not None:
                self.port(call, "scales", "input", dispatch_output.hidden_states_scale, rows)

            result = original(dispatch_output)
            self.port(call, "hidden_states", "output", result.hidden_states, rows)
            return result

    def port(self, call, name, direction, tensor, rows) -> None:
        owner = self.observation
        with owner.tensors.metadata():
            if tensor.ndim == 0 or tensor.shape[0] != call.record["parameters"][rows]:
                raise ValueError(f"native expert tensor disagrees with conditional route geometry: {name}")
            shape = [rows, *tensor.shape[1:]]
            call.port(name, direction, owner.tensors.observe(tensor, shape), use="device")


class NativeExpertRoutes:
    def __init__(self, description, aliases, *, metadata=nullcontext, auxiliary=nullcontext) -> None:
        self.metadata, self.auxiliary = metadata, auxiliary
        self.input = description["serving_input"]
        settings = self.input["configuration"]
        topology = settings["ep"]
        self.size = topology["size"]
        self.source = self.input["ep_routes"][topology["id"]]

        self.aliases = aliases

        self.members: list[dict[str, Any]] | None = None

    def configure(self, experts: dict, layers: list[dict], model_layers: int) -> None:
        get_moe_ep_group = resolve(symbols().get_moe_ep_group)
        get_parallel = resolve(symbols().get_parallel)

        if any(module.moe_ep_size != self.size for module in experts.values()):
            raise ValueError("injected routes must match the native expert group")

        self.conditions = ExpertRoutes(
            self.source, self.input["requests"], layers, model_layers, get_parallel().attn_tp_size - 1
        )
        args = resolve(symbols().get_global_server_args)()
        self.geometry = ExpertBatch(
            self.conditions, self.size, get_moe_ep_group().rank_in_group, args.moe_a2a_backend, args.deepep_mode
        )

        # A module pre-hook runs before the observation wrapper snapshots the expert's input ports.
        for module in experts.values():
            module.register_forward_pre_hook(self.expert, with_kwargs=True)

    def run_batch(self, original: Any, scheduler: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:

        # Representative warmup and capture share the supplied logical choices.
        if not batch.reqs or batch.reqs[0].rid not in self.aliases:
            return original(scheduler, batch, *args, **kwargs)

        previous = self.members
        with self.metadata():
            self.members = BatchTensors.members(batch)
            for member in self.members:
                member["id"] = self.aliases[member["id"]]

        try:
            return original(scheduler, batch, *args, **kwargs)
        finally:
            self.members = previous

    def expert(self, experts: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.members is None:
            return

        result = args[1] if len(args) > 1 else kwargs["topk_output"]
        if result.topk_ids.shape[1] != self.conditions.layers[experts.layer_id]["top_k"]:
            raise ValueError("native expert input does not match the supplied route top-k")

        # DeepEP partitions padded tokens; a local expert group retains the entire token vector.
        with self.auxiliary():
            group = resolve(symbols().get_moe_ep_group)()
            rows = self.conditions.batch_rows(
                self.members, experts.layer_id, result.topk_ids.shape[0] * group.world_size
            )

            selected = torch.tensor(rows, dtype=result.topk_ids.dtype, device="cpu").tensor_split(group.world_size)[
                group.rank_in_group
            ]
            selected = selected.to(result.topk_ids.device)
            location = (
                resolve(symbols().expert_location_dispatch_info).init_new(experts.layer_id)
                if resolve(symbols().get_global_server_args)().ep_dispatch_algorithm is not None
                else None
            )

            if location is not None:
                selected = resolve(symbols().topk_ids_logical_to_physical)(selected, location)
            result.topk_ids.copy_(selected)

    def install(self) -> None:
        HookTable.around(symbols().scheduler_run_batch, self.run_batch)


class NativeContent(RoutedCacheKeys):
    def __init__(self, description, auxiliary=nullcontext) -> None:
        source = description["content"]
        self.content = WorkloadContent(source["requests"], source["prefixes"])
        self.aliases = {
            native: request["id"]
            for native, request in zip(description["source_requests"], source["requests"], strict=True)
        }
        self.execution_aliases = {
            native: identity for phase in description["request_groups"]
            for native, identity in zip(description[f"{phase}_requests"], self.aliases.values(), strict=True)
        }
        self.auxiliary = auxiliary
        self.tokens = None
        self.routing = None
        self.route_assignments = None
        serving = description["serving_input"]

        if "ep" in serving["configuration"]:
            settings = serving["configuration"]
            if settings.get("prefix_cache", False):
                routes = serving["ep_routes"][settings["ep"]["id"]]
                self.route_assignments = routes["assignments"]

    def configure(self, workers) -> None:
        if self.route_assignments is not None:

            # Cache identity covers the whole model, including expert layers on other PP stages.
            layers = {layer["layer"] for worker in workers for layer in worker["expert_layers"]}
            self.routing = RoutedContent(self.content, self.route_assignments, layers)

    def request(self, original, req, *args, **kwargs):
        original(req, *args, **kwargs)
        if req.rid in self.execution_aliases:
            req.extra_key = self.execution_aliases[req.rid]

    def run_batch(self, original, scheduler, batch, *args, **kwargs):
        previous = self.tokens
        aliases = self.execution_aliases
        self.tokens = (
            [self.content.output_token(aliases[req.rid]) for req in batch.reqs]
            if batch.reqs and batch.reqs[0].rid in aliases
            else None
        )

        try:
            return original(scheduler, batch, *args, **kwargs)
        finally:
            self.tokens = previous

    def sample(self, original, runner, *args, **kwargs):
        result = original(runner, *args, **kwargs)
        if self.tokens is not None:

            # Keep the sampler's tensor identity and work; injected values are workload setup, not model work.
            with self.auxiliary():
                result.copy_(torch.tensor(self.tokens, dtype=result.dtype, device=result.device))
        return result

    def install(self) -> None:
        HookTable.around(symbols().scheduler_run_batch, self.run_batch)
        HookTable.around(symbols().model_runner_sample, self.sample)
        if self.route_assignments is not None:
            HookTable.around(symbols().req_init, self.request)
            self.install_cache_keys()
