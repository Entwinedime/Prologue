"""Observe routed expert operands and native dispatch and combine boundaries."""

from functools import partial

from prologue.integrations.frameworks.sglang.expert_routes import ExpertRoutes
from prologue.integrations.frameworks.sglang.observation.parallel import ObservedWork


class ExpertObservation:
    def __init__(self, observation, specification):
        self.observation = observation
        self.specification = specification
        self.width = specification["width"]
        self.members = None
        self.bindings = None
        self.exchange = None

    def loaded(self, runner, experts):
        from sglang.srt.distributed import get_moe_ep_group

        self.group = get_moe_ep_group()
        if self.group.world_size != self.width:
            raise ValueError("the native expert group does not match the workload route table")
        geometry = {module.layer_id: (module.num_experts, module.top_k) for _, module in experts}
        if len(geometry) != len(experts) or any(module.num_fused_shared_experts for _, module in experts):
            raise ValueError("declared routes require one routed expert module per layer and separate shared experts")

        self.table = ExpertRoutes(self.specification["routes"], self.specification["requests"], geometry,
                                  runner.model_config.hf_config.num_hidden_layers)
        self.table.require_geometry(geometry)
        for name, module in experts:
            module.forward = partial(self.forward, module.forward, module, name)
            if self.observation.mode == "detailed":
                module.run_moe_core = partial(self.compute, module.run_moe_core, name)

    def batch(self, original, scheduler, batch, *args, **kwargs):
        aliases = self.observation.requests
        if batch.reqs and batch.reqs[0].rid in self.specification["warmup"]:
            aliases = self.specification["warmup"]
        if not batch.reqs or batch.reqs[0].rid not in aliases:
            return original(scheduler, batch, *args, **kwargs)

        self.members = self.table.members(batch, aliases)
        self.bindings = []
        try:
            return original(scheduler, batch, *args, **kwargs)
        finally:
            self.members = self.bindings = None

    def bind(self, module, topk):
        import torch

        rows, top_k = topk.shape
        if top_k != module.top_k:
            raise ValueError("native routed expert input does not match the declared top-k")
        binding = self.table.bind(self.members, module.layer_id, rows * self.width)
        begin = rows * self.group.rank_in_group
        selected = self.table.shard(binding, begin, begin + rows)
        topk.copy_(torch.tensor(selected, dtype=topk.dtype, device="cpu").reshape(rows, top_k).to(topk.device))
        self.bindings.append(binding)

    def forward(self, original, module, site, hidden_states, topk_output):
        if self.members is None:
            return original(hidden_states, topk_output)
        calls = self.observation.calls
        topk = topk_output.topk_ids
        if calls is None or not calls.stack:
            self.bind(module, topk)
            return original(hidden_states, topk_output)

        # The workload selects experts explicitly, so this copy is capture work outside the native graph.
        with calls.scope("capture_auxiliary", "ExpertRoutes.bind", site, {}):
            self.bind(module, topk)
        with calls.scope("experts", type(module).__qualname__, site, {"layer": module.layer_id}) as call:
            call["inputs"] = calls.ports(hidden=hidden_states, expert_ids=topk)
            result = original(hidden_states, topk_output)
            call["outputs"] = calls.ports(hidden=result)
            return result

    def compute(self, original, site, dispatch_output):
        calls = self.observation.calls
        if not calls.stack:
            return original(dispatch_output)

        parameters = {}
        if self.width > 1:
            parameters["received_per_expert"] = dispatch_output.num_recv_tokens_per_expert
        with calls.scope("expert_compute", "run_moe_core", site, parameters) as call:
            call["inputs"] = calls.ports(hidden=dispatch_output.hidden_states)
            result = original(dispatch_output)
            call["outputs"] = calls.ports(hidden=result.hidden_states)
            return result

    def buffer(self, phase, original, buffer, *args, **kwargs):
        if not self.observation.calls.stack:
            return original(buffer, *args, **kwargs)
        self.exchange = phase
        try:
            return original(buffer, *args, **kwargs)
        finally:
            self.exchange = None

    def collective(self, name, signature, original, *args, **kwargs):
        if self.exchange is None:
            return original(*args, **kwargs)
        values = signature.bind(*args, **kwargs).arguments
        all_to_all = name == "all_to_all_single"
        source, target = (values["input"], values["output"]) if all_to_all else (
            values["input_tensor"], values["output_tensor"])
        parameters = {"kind": "all_to_all" if all_to_all else "all_gather", "participants": self.group.ranks,
                      "phase": self.exchange}
        if all_to_all:
            parameters.update(send_rows=values["input_split_sizes"], receive_rows=values["output_split_sizes"])
        else:
            parameters["dimension"] = 0
        calls = self.observation.calls
        with calls.scope("collective", f"distributed.{name}", "expert_parallel", parameters) as call:
            call["inputs"] = calls.ports(operand=source)
            result = original(*args, **kwargs)
            call["outputs"] = calls.ports(result=target)
        if values.get("async_op"):
            return ObservedWork(self.observation, result, "collective_wait", "expert_parallel", call["id"])
        return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register("sglang.srt.managers.scheduler.Scheduler.run_batch", self.batch, HookType.AROUND)
        if self.width == 1 or self.observation.mode != "detailed":
            return

        from inspect import signature

        import torch.distributed as distributed

        for method, phase in (("get_dispatch_layout", "layout"), ("dispatch", "dispatch"), ("combine", "combine")):
            HookRegistry.register(f"deep_ep.Buffer.{method}", partial(self.buffer, phase), HookType.AROUND)
        for name in ("all_gather_into_tensor", "all_to_all_single"):
            hook = partial(self.collective, name, signature(getattr(distributed, name)))
            HookRegistry.register(f"torch.distributed.{name}", hook, HookType.AROUND)
