"""Observe the same batch, model and sampling boundaries in serving and local execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.call_scopes import NativeCallObservation
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors
from prologue.integrations.frameworks.sglang.shared.pipeline import PipelineABI


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
        if not self.active:
            return original(scheduler, batch, *args, **kwargs)
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
        if not self.active:
            return original(runner, logits_output, forward_batch)
        with self.call("sample", original, device=True) as call:
            self.logits_port(call, "input", logits_output.next_token_logits)
            tokens = original(runner, logits_output, forward_batch)
            self.token_output(call, tokens)
            return tokens

    def forward(self, original: Any, runner: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.active:
            return original(runner, batch, *args, **kwargs)
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
