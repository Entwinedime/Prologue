"""Keep native layer weights on the host when several logical workers share one device."""

from collections import defaultdict
from functools import partial, update_wrapper
from itertools import chain

import torch
import torch.distributed as distributed

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes


class LayerWeights:
    def __init__(self, tensors, auxiliary):
        self.auxiliary = auxiliary
        self.storage = [(tensor, torch.empty(tensor.untyped_storage().nbytes(), dtype=torch.uint8, device="cpu"))
                        for tensor in tensors]

    def move(self, to_device):
        with self.auxiliary():
            if not to_device:

                # Kernels on other native streams can still be reading this layer after forward returns.
                device_runtime().api.synchronize()
            for tensor, backing in self.storage:
                device_runtime().move_weight(tensor, backing, to_device)

    def forward(self, original, *args, **kwargs):
        self.move(True)
        try:
            return original(*args, **kwargs)
        finally:
            self.move(False)


class ShardWeights:
    def __init__(self, directory, workers, auxiliary):
        self.directory, self.workers, self.auxiliary = directory, workers, auxiliary

    def measure(self, original, runner, *args, **kwargs):
        available_gib = original(runner, *args, **kwargs)
        self.share = int(available_gib * (1 << 30) * runner.mem_fraction_static) // self.workers
        return available_gib

    def load(self, original, *args, **kwargs):
        model = original(*args, **kwargs)
        weights = {
            t.untyped_storage()._cdata: t
            for t in model.parameters()
            if t.device.type == platform_domain().device_type and t.numel()
        }
        footprint = sum(t.untyped_storage().nbytes() for t in weights.values())
        if footprint <= self.share:
            return model

        modules = dict(model.named_modules())
        layers = {name for name, role in ModelScopes.selected(modules).items() if role == "layer"}
        owners = defaultdict(set)
        for path, tensor in chain(
            model.named_parameters(remove_duplicate=False), model.named_buffers(remove_duplicate=False)
        ):
            container = path.rpartition(".")[0]
            while container and container not in layers:
                container = container.rpartition(".")[0]
            owners[tensor.untyped_storage()._cdata].add(container)

        # Shared embeddings, cross-layer aliases and buffers outside a layer stay available to their actual consumers.
        groups = defaultdict(list)
        for key, tensor in weights.items():
            if len(owners[key]) == 1:
                owner, = owners[key]
                if owner in layers:
                    groups[owner].append(tensor)

        device_runtime().load_weight_bridge()
        parked = 0
        for name, tensors in groups.items():
            residence = LayerWeights(tensors, self.auxiliary)
            residence.move(False)
            layer = modules[name]
            layer.forward = update_wrapper(partial(residence.forward, layer.forward), layer.forward)
            parked += sum(backing.nbytes for _, backing in residence.storage)
        write_json(
            self.directory / "weight-residency" / f"{distributed.get_rank()}.json",
            {
                "native_weight_bytes": footprint,
                "worker_share_bytes": self.share,
                "parked_bytes": parked,
                "layers": len(groups),
            },
        )
        return model
