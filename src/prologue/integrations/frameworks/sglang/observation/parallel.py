"""Capture native worker roles and their logical tensor collectives."""

import json
from functools import partial
from pathlib import Path


class RankProfiles:
    def __init__(self, observation):
        self.observation = observation
        self.root = observation.output
        self.selected = False

    def loaded(self, runner):
        from sglang.srt.distributed import get_tp_group

        group = get_tp_group()
        rank = runner.tp_rank
        model = runner.model
        heads = sorted({(layer.self_attn.attn.tp_q_head_num, layer.self_attn.attn.tp_k_head_num,
                         layer.self_attn.attn.tp_v_head_num) for layer in model.model.layers})
        layout = {
            "publication_owner": rank == 0, "participants": group.ranks, "heads": heads,
            "modules": [(name, type(module).__qualname__) for name, module in model.named_modules()],
            "parameters": [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in model.named_parameters()],
        }
        (self.root / f"layout-{rank}.json").write_text(json.dumps(layout), encoding="utf-8")

    def profile(self, original, manager, request):
        from sglang.srt.managers.io_struct import ProfileReqOutput, ProfileReqType
        from torch.distributed import barrier

        rank = manager.ps.tp_rank
        starting = request.req_type == ProfileReqType.START_PROFILE
        if starting:
            window = Path(request.output_dir)
            selection = json.loads((window / "selection.json").read_text(encoding="utf-8"))
            self.selected = rank in selection["ranks"]
            output = window / f"rank-{rank}"
            self.observation.begin_window(output, selection["requests"], self.selected)
            request.output_dir = str(output / "profile")
        elif self.selected or rank == 0:
            self.observation.save()

        if self.selected:
            result = original(manager, request)
        else:
            if not starting:

                # Exporting ranks join this same native stop barrier after closing their trace.
                barrier(manager.dp_tp_cpu_group)
            result = ProfileReqOutput(success=True, message="The worker participates without a device profile")

        return result

    def collective(self, name, original, group, input_, *args, **kwargs):
        calls = self.observation.calls
        if not calls.stack or group.world_size == 1 or input_.device.type != "npu":
            return original(group, input_, *args, **kwargs)

        parameters = {"kind": name, "participants": group.ranks}
        if name == "all_reduce":
            parameters["reduction"] = "sum"
        else:
            parameters["dimension"] = args[0] if args else kwargs.get("dim", -1)
        with calls.scope("collective", f"GroupCoordinator.{name}", "tensor_parallel", parameters) as call:
            call["inputs"] = calls.ports(operand=input_)
            result = original(group, input_, *args, **kwargs)
            call["outputs"] = calls.ports(result=result)
            return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        manager = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager._profile"
        HookRegistry.register(manager, self.profile, HookType.AROUND)
        group = "sglang.srt.distributed.parallel_state.GroupCoordinator."
        for name in ("all_reduce", "all_gather"):
            HookRegistry.register(group + name, partial(self.collective, name), HookType.AROUND)


class RankPlan:
    def __init__(self, directory, width, placement):
        self.workers = {}
        representatives = {}
        for rank in range(width):
            layout = json.loads((directory / f"layout-{rank}.json").read_text(encoding="utf-8"))
            if layout["participants"] != list(range(width)):
                raise ValueError("native TP membership differs from the requested topology")
            key = json.dumps(layout, sort_keys=True)
            representative = representatives.setdefault(key, rank)
            self.workers[rank] = {
                "representative": representative, "publication_owner": layout["publication_owner"],
                "heads": layout["heads"],
            }
        if placement == "single_device":
            self.windows = [[rank] for rank in representatives.values()]
        else:
            self.windows = [list(self.workers)]
