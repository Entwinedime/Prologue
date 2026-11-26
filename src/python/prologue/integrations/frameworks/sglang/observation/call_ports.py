"""Record native call boundaries, operands and scope parameters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from functools import update_wrapper
from typing import Any

import torch

from prologue.common.observation import SemanticCapture
from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestStateCalls
from prologue.integrations.frameworks.sglang.shared.calls import native_implementation
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI
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
