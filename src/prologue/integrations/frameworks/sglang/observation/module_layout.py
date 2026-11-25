"""Read initialized native module structure and layer communication choices."""

from __future__ import annotations

from functools import partial, update_wrapper
from inspect import signature
from typing import Any


class LayerCommunication:
    def __init__(self, native) -> None:
        self.native = native

    @classmethod
    def read(cls, module):
        from sglang.srt.layers.communicator import LayerCommunicator

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
            from sglang.srt.layers.communicator import get_attn_tp_context

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
        from sglang.srt.layers.utils.common import PPMissingLayer

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
