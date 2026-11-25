"""Supply synthetic output content without replacing native sampling work."""

from contextlib import nullcontext

import torch

from prologue.common.workload import WorkloadContent
from prologue.common.workload.routing import RoutedContent
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.shared.cache_keys import RoutedCacheKeys


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
