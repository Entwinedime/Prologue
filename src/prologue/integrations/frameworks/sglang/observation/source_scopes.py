"""Lower native resource and shape declarations for projection and source checking."""

from prologue.integrations.frameworks.sglang.batches import BatchTensors
from prologue.integrations.frameworks.sglang.resources import ResourceInputs


class SGLangSourceScopes:
    relay_ports = {"run_batch": "output", "output_prepare": "input"}
    qwen_layer = "sglang.srt.models.qwen3.Qwen3DecoderLayer"
    rotary_layers = {
        qwen_layer,
        "sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer",
        "sglang.srt.models.llama.LlamaDecoderLayer",
    }
    packed_value_layouts = {
        "sglang.srt.models.llama.LlamaDecoderLayer": {"prefill", "mixed"},
        "sglang.srt.models.qwen2_moe.Qwen2MoeDecoderLayer": {"decode"},
    }
    model_calls = {"launch", "run_batch", "model"}
    prebuilt_batches = {"prebuilt_prepare", "prebuilt_commit", "prebuilt_result"}

    def __init__(self, environment):
        self.environment = environment
        self.backend = environment["implementation"]["attention_backend"]
        self.first_layer = str(environment["parallel"]["layer_range"][0])

    def layout_operand(self, scope, scopes):
        direction = self.relay_ports.get(scope["role"])
        if direction and any(access["field"] == "output_token" and access["access"] == "write"
                             for access in scope.get("state_accesses", ())):

            # FutureMap writes these values through Python indexed assignment, which strips leading unit axes.
            return "next_token_ids", direction
        if (scope["implementation"] == "sglang.srt.mem_cache.common.alloc_for_decode"
                and scope["role"] == "segment" and scope["work"]["continuation"]["exit"] == "call.return"):
            return "cache_locations", "output"
        role = scope["role"]
        if role not in {"layer", "attention"} or not self.environment["device"].startswith("Ascend"):
            return None

        phase = scope["work"]["phase"]
        if role == "attention":
            layer = scopes[scope["parent"]]["implementation"]
            if self.backend == "ascend" and phase in self.packed_value_layouts.get(layer, ()):

                # The native packed value slice materializes its row gap only for a multirow cache write.
                return "argument.v", "input"
            return None

        implementation = scope["implementation"]
        if implementation == self.qwen_layer and phase in {"prefill", "mixed"}:

            # Flattening each packed Q/K view for normalization copies when it crosses token rows.
            return "argument.positions", "input"
        if (implementation in self.rotary_layers
                and scope["site"].rsplit(".", 1)[-1] == self.first_layer and phase == "decode"):

            # The first local decoder makes the strided RoPE rows contiguous only when there are multiple positions.
            return "argument.positions", "input"
        return None

    def lower(self, scopes):
        result = []
        for scope in ResourceInputs.scopes(scopes, self.environment):
            guards = list(scope.get("guards", ()))

            # Native prebuilt batches prepare KV indices, relay tokens and finish requests one member at a time.
            if scope["bindings"].get("segment", {}).get("owner") in self.prebuilt_batches:
                guards.append(["eq", "members", scope["parameters"]["members"]])
            if scope["role"] in self.model_calls:
                layout = BatchTensors.attention_layout(self.backend, scope["parameters"], scope["work"]["members"])
                if layout:
                    scope = {**scope, "parameters": scope["parameters"] | layout}
                    if scopes[scope["parent"]]["role"] in self.model_calls:
                        scope["arguments"] = scope["arguments"] | {name: name for name in layout}
                if scope["role"] == "model":
                    guards.extend(["eq", name, value] for name, value in layout.items())
                    guards.extend(BatchTensors.model_padding_guards(scope["parameters"]))

            operand = self.layout_operand(scope, scopes)
            if operand is not None:
                tensor, = (port["tensor"] for port in scope["ports"] if (port["name"], port["direction"]) == operand)
                size = tensor["shape"][0]
                extent = tensor["shape_rule"][0]
                guards.append(["eq", extent, 1] if size == 1 else ["lt", 1, extent])
            result.append({**scope, "guards": guards} if guards else scope)
        return result
