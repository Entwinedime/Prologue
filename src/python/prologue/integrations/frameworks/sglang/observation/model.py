"""Observe native model calls, module cuts, expert routes and tensor ports."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import fields, is_dataclass
from functools import partial, update_wrapper
from typing import Any

import torch

from prologue.common.observation import SemanticCapture
from prologue.common.workload import ExpertRoutes, RoutedContent, WorkloadContent
from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.layout import LayerCommunication, ModuleStructure
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestStateCalls, RoutedCacheKeys
from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI, native_implementation
from prologue.integrations.frameworks.sglang.shared.configuration import ExpertBatch, MhaKvLayout
from prologue.integrations.runtimes.pytorch.capture import TensorBoundaries


class LocalModel:
    def __init__(self, domain, page_size, **options):
        impl = domain["implementation"]
        self.args = resolve(symbols().server_args)(
            model_path=impl["model"]["_name_or_path"], dtype=impl["dtype"].removeprefix("torch."),
            device=platform_domain().device_type, tp_size=domain["parallel"]["logical_tp"], page_size=page_size,
            attention_backend=impl["attention_backend"], load_format=impl["load_format"], skip_tokenizer_init=True,
            disable_cuda_graph=True, disable_overlap_schedule=True, **options,
        )
        resolve(symbols().set_global_server_args_for_scheduler)(self.args)
        self.config = resolve(symbols().model_config).from_server_args(self.args)
        self.document = json.loads(self.config.hf_config.to_json_string(use_diff=False))
        self.matches = self.document == impl["model"] and str(self.config.dtype) == impl["dtype"]


def enabled_hook(observation, callback):
    def invoke(original, *args, **kwargs):
        if observation.enabled:
            return callback(original, *args, **kwargs)
        return original(*args, **kwargs)

    return update_wrapper(invoke, callback)


class NativeCallObservation:
    def __init__(self, markers, mode, *, resource_trace=False):
        self.mode = mode
        self.capture = SemanticCapture(markers)
        self.tensors = TensorBoundaries(self.capture)
        self.request_state = RequestStateCalls(observe=resource_trace)
        self.action_values = []
        self.selected = True
        self.active = False
        self.resource_history = None

    @property
    def enabled(self):
        return self.active

    def document(self, environment, state_queries):
        raw = self.capture.document(environment)
        raw.update(self.capture.runtime.timing_evidence(), state_queries=state_queries)
        return raw

    @property
    def protocol_enabled(self):
        return self.active

    @contextmanager
    def call(self, role: str, original: Any, *, parameters=None, arguments=None, work=None, device=False,
             stream=None, implementation=None):
        outer = self.capture.current.get() == 0
        try:
            with ExitStack() as contexts:
                call = contexts.enter_context(self.capture.call(
                    role, original.__qualname__, implementation or native_implementation(original),
                    parameters=parameters, arguments=arguments, work=work, device=device, stream=stream,
                    constants=None if parameters is None else {
                        name: parameters[name] for name in ("page_size", "sequence_capacity") if name in parameters
                    },
                ))
                if outer:
                    contexts.enter_context(self.tensors)
                with self.request_state.record() as accesses:
                    yield call
                for key, requests in accesses.members.items():
                    identities = [self.aliases[request] for request in requests]
                    if not identities:
                        continue
                    field, mode = key.split("/")
                    call.record["work"][key] = identities
                    call.record.setdefault("state_accesses", []).append({
                        "space": "slot", "pool": "local", "field": field, "access": mode,
                        "members": key, "requests": identities, "interval": "device",
                    })
            if accesses.values:
                self.action_values.append({"scope": call.identity, "values": {"state": [
                    {**value, "request": self.aliases[value["request"]]} for value in accesses.values
                ]}})
        finally:
            if outer:
                self.tensors.retire()


class NativeModelObservation(NativeCallObservation):
    def batch_work(self, batch: Any) -> dict[str, Any]:
        return self.calls.batch_work(batch)

    @contextmanager
    def model_call(self, role, original, batch, *, stream=None):
        work = self.batch_work(batch)
        parameters = BatchTensors.parameters(batch, sum(member["new_tokens"] for member in work["members"]))
        parameters.update(BatchTensors.model_parameters(batch, parameters["input_tokens"]))
        bindings = self.routing.geometry.bind_batch(parameters, work["members"]) if self.routing is not None else {}

        with self.call(role, original, parameters=parameters, work=work, device=True, stream=stream) as call:
            yield call

            # Children inherit batch inputs while running. Only the model call owns the peer bindings.
            call.record["work"].update(bindings)

    def run_batch(self, original: Any, scheduler: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        with self.request_state.members(batch), self.model_call("run_batch", original, batch) as call:
            if self.resource_history is not None:
                self.resource_history.batch(call.record["id"], batch)
            parent = self.capture.scopes[call.record["parent"]]
            if parent["role"] == "launch":

                # PP launch and its run_batch child share the same declared batch ABI.
                call.record["arguments"] = {
                    name: name for name in call.record["parameters"] if name not in call.record["constants"]
                }
            result = original(scheduler, batch, *args, **kwargs)
            self.result_outputs(call, result)
            return result

    def batch_ports(
        self, call: Any, batch: Any, direction: str,
        *, names=("request_slots", "sequence_lengths", "cache_locations"),
    ) -> None:
        columns = BatchTensors.boundaries(batch)
        for name in names:
            tensor, shape = columns[name]
            call.port(name, direction, self.tensors.observe(tensor, shape), use="device")

    def logits_port(self, call, direction, logits):
        # The runner hands the sampler one unpadded vocabulary row per admitted request.
        call.port(
            "next_token_logits", direction, self.tensors.observe(logits, ["members", logits.shape[1]]), use="device"
        )

    def token_output(self, call, tokens):
        use = "host" if tokens.device.type == "cpu" else "device"
        call.port("next_token_ids", "output", self.tensors.observe(tokens, ["members"]), use=use)

    def sample(self, original: Any, runner: Any, logits_output: Any, forward_batch: Any) -> Any:
        with self.call("sample", original, device=True) as call:
            self.logits_port(call, "input", logits_output.next_token_logits)
            tokens = original(runner, logits_output, forward_batch)
            self.token_output(call, tokens)
            return tokens

    def forward(self, original: Any, runner: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        LogitsProcessorOutput = resolve(symbols().logits_processor_output)

        # The initialized native registry owns these allocations across calls; a module path does not.
        with self.tensors.metadata():
            registry = runner.eager_runner._eager_registry
            owners = {}
            for name in sorted(registry.slot_names()):
                tensor = registry.get_slot(name).buffer
                if tensor is not None:
                    owner = owners.setdefault(tensor.untyped_storage()._cdata, f"eager/{name}")
                    self.tensors.bind_buffer(tensor, owner)
        with self.call("model", original, device=True) as call:
            call.record["parameters"].update(BatchTensors.input_dimensions(batch.input_ids.numel(), batch.batch_size))
            self.batch_ports(call, batch, "input")
            result = original(runner, batch, *args, **kwargs)
            output = result.logits_output
            if isinstance(output, LogitsProcessorOutput) and output.next_token_logits is not None:
                self.logits_port(call, "output", output.next_token_logits)
        self.capture.runtime.checkpoint()
        return result

    def model_geometry(self, dimensions: dict[str, int]) -> None:
        # Check actual model tensors against the shared native padding contract, without replacing predicted features.
        parameters = self.capture.scopes[self.capture.current.get()]["parameters"]
        if any(parameters[name] != actual for name, actual in dimensions.items()):
            raise ValueError("native model geometry differs from the declared eager batch layout")

    def result_outputs(self, call: Any, output: Any) -> None:
        tensor = output.next_token_ids
        if tensor is not None:
            self.token_output(call, tensor)

        proxy = output.pp_hidden_states_proxy_tensors
        if proxy is not None:
            for name, tensor in PipelineABI.proxy_tensors(proxy).items():
                rule = self.tensors.shape_rule(tensor)
                call.port(f"proxy.{name}", "output", self.tensors.observe(tensor, rule), use="device")


class AttentionCachePorts:
    def __init__(self, tensors: Any, module: Any, arguments: dict[str, Any]) -> None:

        self.tensors = tensors
        self.layer = module.layer_id
        pool = resolve(symbols().get_attn_backend)().token_to_kv_pool
        local = self.layer - pool.start_layer
        self.layout = MhaKvLayout.from_pool(pool)

        # The native layout gives each layer a disjoint slice even though all layers share an allocation.
        with tensors.metadata():
            self.buffers = {"cache.key": pool.k_buffer[local], "cache.value": pool.v_buffer[local]}
        self.token_bytes = self.layout.key_token_bytes + self.layout.value_token_bytes
        self.region = f"kv_layer/{self.layer}"
        self.writes = arguments.get("save_kv_cache", True) and arguments.get("k") is not None

    def ports(self, call: Any, direction: str) -> None:
        if direction == "output" and not self.writes:
            return
        members = call.record["work"]["members"]
        call.record.setdefault("state_accesses", []).append(
            {
                "space": "kv",
                "pool": "local",
                "layer": self.layer,
                "access": "read" if direction == "input" else "write",
                "interval": "device",
                "rule": {
                    "begin": 0 if direction == "input" else "context_tokens",
                    "end": "context_tokens" if direction == "input" else ["add", "context_tokens", "new_tokens"],
                },
                "slice": {"token_bytes": self.token_bytes, "begin": 0, "end": self.token_bytes},
                "regions": [
                    {
                        "request": member["id"],
                        "begin": 0 if direction == "input" else member["context_tokens"],
                        "end": member["context_tokens"]
                        if direction == "input"
                        else member["context_tokens"] + member["new_tokens"],
                    }
                    for member in members
                ],
            }
        )

        # Layout ports do not imply whole-buffer hazards; readiness is selected by logical request/token regions.
        for name, tensor in self.buffers.items():
            if direction == "output":
                self.tensors.written(tensor, self.region)
            shape = [["add", "kv_pages", 1], "page_size", *tensor.shape[2:]]
            call.port(name, direction, self.tensors.observe(tensor, shape, region=self.region), use="state")


def tensor_ports(value: Any, path: str, proxy_type=None) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    proxy_type = proxy_type or resolve(symbols().pp_proxy_tensors)
    if isinstance(value, proxy_type):
        yield from tensor_ports(value.tensors, path, proxy_type)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensor_ports(getattr(value, field.name), f"{path}.{field.name}", proxy_type)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from tensor_ports(item, f"{path}.{name}", proxy_type)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_ports(item, f"{path}.{index}", proxy_type)


class ModelScopes:
    token_arguments = {
        "input_ids", "positions", "input_embeds", "hidden_states", "residual", "post_residual_addition",
        "captured_last_layer_outputs", "pp_proxy_tensors", "input_", "q", "k", "v",
    }

    @staticmethod
    def install_batch_hooks(observer) -> None:
        hooks = {
            symbols().scheduler_run_batch: observer.run_batch,
            symbols().model_runner_forward: observer.forward,
            symbols().model_runner_sample: observer.sample,
        }
        HookTable.group({target: enabled_hook(observer, callback) for target, callback in hooks.items()}).install()

    @staticmethod
    def selected(modules):
        RadixAttention = resolve(symbols().radix_attention)
        PPMissingLayer = resolve(symbols().pp_missing_layer)
        ParallelLMHead = resolve(symbols().parallel_lm_head)
        VocabParallelEmbedding = resolve(symbols().vocab_parallel_embedding)

        selected = {"": "model_body"}
        containers = [name for name, module in modules.items() if isinstance(module, torch.nn.ModuleList)]
        outer = [name for name in containers
            if not any(name.startswith(parent + ".") for parent in containers if parent != name)]
        for container in outer:
            for name, layer in modules[container].named_children():
                if not isinstance(layer, PPMissingLayer):
                    path = f"{container}.{name}" if container else name
                    selected[path] = "layer"
        for name, module in modules.items():
            if isinstance(module, RadixAttention):
                selected[name] = "attention"
            elif isinstance(module, VocabParallelEmbedding) and not isinstance(module, ParallelLMHead):
                selected[name] = "embedding"
        return selected

    @staticmethod
    def expert_structure(modules):
        FusedMoE = resolve(symbols().fused_mo_e)
        TopK = resolve(symbols().top_k)

        experts = {name: module for name, module in modules.items() if isinstance(module, FusedMoE)}
        layers = []
        for path, module in experts.items():

            # The enclosing MoE block pairs its routing module with the expert executor; executor top_k is optional.
            router, = [child for child in modules[path.rpartition(".")[0]].children() if isinstance(child, TopK)]
            layers.append({
                "layer": module.layer_id, "experts": module._num_global_routed, "top_k": router.topk_config.top_k,
            })
        return experts, layers

    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.bindings = {}

    def attach_model(self, model: Any) -> None:
        LogitsProcessor = resolve(symbols().logits_processor)

        modules = dict(model.named_modules())
        self.experts, self.expert_layers = self.expert_structure(modules)
        experts = set(self.experts.values())
        for module in modules.values():
            if isinstance(module, LogitsProcessor):
                original = module._compute_lm_head
                module._compute_lm_head = update_wrapper(partial(self.logits, original), original)

        for path, role in self.selected(modules).items():
            module = modules[path]
            expert_layers = {child.layer_id for child in module.modules()
                if child in experts} if role == "layer" else set()
            if len(expert_layers) > 1:
                raise ValueError("decoder scope contains multiple expert layers with independent routes")
            self.attach(path, role, module, next(iter(expert_layers), None))

        # A pipeline input enters the first local decoder with that decoder's native token partition.
        first_layer = next(spec for spec in self.bindings.values() if spec["role"] == "layer")
        self.proxy_rows = first_layer["rows"]["input"]

    def attach(self, path, role, module, expert_layer=None):
        original = module.forward
        rows = {"input": "model_tokens", "output": "model_tokens"}
        if role == "layer" and (communicator := LayerCommunication.read(module)) is not None:
            rows = communicator.observe(self.observation)
        self.bindings[path] = {
            "rows": rows, "expert_layer": expert_layer, "role": role, "signature": inspect.signature(original),
            "implementation": f"{type(module).__module__}.{type(module).__qualname__}",
            "weights": [name for name, _ in module.named_parameters(recurse=False)], "module": module,
        }

        # Native capability checks inspect forward's signature after instrumentation is installed.
        module.forward = update_wrapper(partial(self.invoke, path, original), original)

    def logits(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        owner = self.observation
        if owner.active and owner.selected:
            parameters = owner.capture.scopes[owner.capture.current.get()]["parameters"]

            # Serving without input-logprob expansion projects one selected hidden state per model member.
            if result.ndim != 2 or result.shape[0] != parameters["model_members"]:
                raise ValueError("logits projection needs a declared row relation for this native path")
            with owner.tensors.metadata():
                owner.tensors.declare_shape(result, ["model_members", result.shape[1]])
        return result

    def shape(self, spec, name: str, tensor: torch.Tensor, *, output: bool) -> list[Any] | None:
        if not output and name.split(".", 2)[1] not in self.token_arguments:
            return self.observation.tensors.declared_shape(tensor)
        if output and spec["role"] == "model_body":

            # The model wrapper forwards layer/logits results; it cannot replace their producer geometry.
            declared = self.observation.tensors.declared_shape(tensor)
            if declared is not None:
                return declared

        shape: list[Any] = list(tensor.shape)
        if not shape:
            return shape

        # The decoder ABI is flat-token based; logits select one row per request, independently of token count.
        axis = "model_members" if output and ".next_token_" in name else "model_tokens"
        if not output and name.startswith("argument.pp_proxy_tensors."):
            axis = self.proxy_rows
        if spec["role"] == "layer" and (output
            or name.split(".", 2)[1] in {"hidden_states", "residual", "post_residual_addition"}):
            axis = spec["rows"]["output" if output else "input"]
        shape[0] = axis
        return shape

    def proxy_shape(self, tensor):
        return [self.proxy_rows, *tensor.shape[1:]]

    @staticmethod
    def bound_arguments(signature, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    @staticmethod
    def model_dimensions(arguments):
        tokens = arguments["input_ids"]
        if tokens is None:
            tokens = arguments["pp_proxy_tensors"].tensors["hidden_states"]
        return {"model_tokens": int(tokens.shape[0]), "model_members": int(arguments["forward_batch"].batch_size)}

    def invoke(self, path: str, original: Any, *args: Any, **kwargs: Any) -> Any:
        owner = self.observation
        if not owner.active or not owner.selected:
            return original(*args, **kwargs)

        spec = self.bindings[path]
        arguments = self.bound_arguments(spec["signature"], args, kwargs)
        if spec["role"] == "model_body":
            owner.model_geometry(self.model_dimensions(arguments))
        inputs = []
        for name, value in arguments.items():
            if name == "forward_batch":
                continue
            inputs.extend(tensor_ports(value, f"argument.{name}"))

        cache = None
        if spec["role"] == "attention":

            cache = AttentionCachePorts(owner.tensors, spec["module"], arguments)
            if cache.layout != owner.kv_layout:
                raise NotImplementedError("multiple KV pool layouts require separate pool identities")

        scope = {}
        if spec["role"] == "embedding":
            tokens = arguments["input_"]
            scope = {"parameters": {"model_tokens": int(tokens.shape[0])},
                     "arguments": {"model_tokens": "model_tokens"} if owner.capture.current.get() else {}}
        elif spec["role"] == "layer" and owner.routing is not None:
            enclosing = owner.capture.scopes[owner.capture.current.get()]
            parameters, relations = owner.routing.geometry.scope(enclosing, spec["expert_layer"])
            scope = {"parameters": parameters, "arguments": relations, "constants": enclosing["constants"]}
        with owner.capture.call(
            spec["role"], path, spec["implementation"],
            bindings={"weights": spec["weights"]}, device=True, **scope,
        ) as call:
            self.ports(call, spec, "input", inputs, cache)
            result = original(*args, **kwargs)
            self.ports(call, spec, "output", tensor_ports(result, "result"), cache)
            return result

    def ports(self, call, spec, direction, tensors, cache):
        if cache is not None:
            cache.ports(call, direction)
        for name, tensor in tensors:
            rule = self.shape(spec, name, tensor, output=direction == "output")
            use = "device" if tensor.device.type == platform_domain().device_type else "host"
            call.port(name, direction, self.observation.tensors.observe(tensor, rule), use=use)


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
