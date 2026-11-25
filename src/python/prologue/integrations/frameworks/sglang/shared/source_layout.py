"""Lower native resource and shape declarations for projection and source checking."""

from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors
from prologue.integrations.frameworks.sglang.shared.resources import ResourceInputs


class SourceLayout:
    relay_ports = {"run_batch": "output", "output_prepare": "input"}

    model_calls = {"launch", "run_batch", "model"}
    prebuilt_batches = {"prebuilt_prepare", "prebuilt_commit", "prebuilt_result"}

    def __init__(self, environment, platform_layouts):
        # Each entry comes from the native decoder's layout at its observation boundary.
        self.position_layouts = {
            symbols().qwen3_decoder_layer: {"prefill", "mixed", "first_decode"},
            symbols().qwen3_moe_decoder_layer: {"first_decode"},
            symbols().llama_decoder_layer: {"first_decode"},
        }
        self.packed_value_layouts = {
            symbols().llama_decoder_layer: {"prefill", "mixed"},
            symbols().qwen2_moe_decoder_layer: {"decode"},
        }
        self.layouts = platform_layouts(environment)
        self.environment = environment
        self.backend = environment["implementation"]["attention_backend"]
        self.first_layer = str(environment["parallel"]["layer_range"][0])

    def layout_operand(self, scope, scopes):
        direction = self.relay_ports.get(scope["role"])
        if direction and any(access["field"] == "output_token" and access["access"] == "write"
                             for access in scope.get("state_accesses", ())):

            # FutureMap writes these values through Python indexed assignment, which strips leading unit axes.
            return "next_token_ids", direction
        if (scope["implementation"] == symbols().alloc_for_decode
                and scope["role"] == "segment" and scope["work"]["continuation"]["exit"] == "call.return"):
            return "cache_locations", "output"
        role = scope["role"]
        if role not in {"layer", "attention"} or not self.layouts:
            return None

        phase = scope["work"]["phase"]
        if role == "attention":
            layer = scopes[scope["parent"]]["implementation"]
            if "packed_value" in self.layouts and phase in self.packed_value_layouts.get(layer, ()):

                # The native packed value slice materializes its row gap only for a multirow cache write.
                return "argument.v", "input"
            return None

        local_phase = phase
        if phase == "decode" and scope["site"].rsplit(".", 1)[-1] == self.first_layer:
            local_phase = "first_decode"
        if "positions" in self.layouts and local_phase in self.position_layouts.get(scope["implementation"], ()):
            return "argument.positions", "input"
        return None

    def lower(self, scopes):
        result = []
        for scope in ResourceInputs.scopes(scopes, self.environment):
            guards = list(scope.get("guards", ()))
            if "page_table" in scope:
                table = scope["page_table"]
                BatchTensors.bind_location_rows(scope, table["slots"], table["writes"])

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
