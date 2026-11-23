"""Capture native worker roles and their local tensor collectives."""

import json
from copy import copy
from functools import partial
from pathlib import Path


class RankProfiles:
    def __init__(self, observation):
        self.observation = observation
        self.root = observation.output
        self.selected = False
        self.window = None

    def channels(self, original, scheduler, port_args):
        result = original(scheduler, port_args)
        self.publication_owner = scheduler.ipc_channels.send_to_detokenizer.socket is not None
        return result

    def loaded(self, runner):
        from sglang.srt.distributed import get_pp_group, get_tp_group

        tp = get_tp_group()
        pp = get_pp_group()
        model = runner.model
        start, end = model.model.start_layer, model.model.end_layer
        heads = sorted({(layer.self_attn.attn.tp_q_head_num, layer.self_attn.attn.tp_k_head_num,
                         layer.self_attn.attn.tp_v_head_num) for layer in model.model.layers[start:end]})
        layout = {
            "publication_owner": self.publication_owner, "tp_leader": tp.is_first_rank,
            "stage": pp.rank_in_group, "layers": [start, end], "heads": heads,
            "modules": [(name, type(module).__qualname__) for name, module in model.named_modules()],
            "parameters": [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in model.named_parameters()],
        }
        workers = {"layout": layout, "tp_members": tp.ranks, "pp_members": pp.ranks}
        (self.root / f"layout-{tp.rank}.json").write_text(json.dumps(workers), encoding="utf-8")

    def profile(self, original, manager, request):
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.managers.io_struct import ProfileReqOutput, ProfileReqType
        from torch.distributed import barrier

        tp = get_tp_group()
        starting = request.req_type == ProfileReqType.START_PROFILE
        if starting:
            window = Path(request.output_dir)
            selection = json.loads((window / "selection.json").read_text(encoding="utf-8"))
            self.selected = tp.rank in selection["ranks"]
            self.stage_profiles = any(rank in selection["ranks"] for rank in tp.ranks)
            self.window = window / f"rank-{tp.rank}"
            self.window.mkdir()
            self.observation.begin_window(self.window, selection["requests"], self.selected)

            # The scheduler forwards the original request to the next stage after this hook.
            request = copy(request)
            request.output_dir = str(self.window / "profile")
        elif self.selected or self.publication_owner:
            self.observation.save()

        if self.selected:
            result = original(manager, request)
        else:
            if not starting and self.stage_profiles:

                # Exporting ranks join this same native stop barrier after closing their trace.
                barrier(manager.dp_tp_cpu_group)
            result = ProfileReqOutput(success=True, message="The worker participates without a device profile")

        if result.success:
            (self.window / ("started" if starting else "stopped")).touch()
        if not starting:
            self.selected = False
            self.observation.capture_calls = False
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

        scheduler = "sglang.srt.managers.scheduler.Scheduler."
        HookRegistry.register(scheduler + "init_ipc_channels", self.channels, HookType.AROUND)
        manager = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager._profile"
        HookRegistry.register(manager, self.profile, HookType.AROUND)
        group = "sglang.srt.distributed.parallel_state.GroupCoordinator."
        for name in ("all_reduce", "all_gather"):
            HookRegistry.register(group + name, partial(self.collective, name), HookType.AROUND)


class RankPlan:
    def __init__(self, directory, configuration, placement):
        self.workers = {}
        representatives = {}
        for rank in range(configuration.worker_count):
            loaded = json.loads((directory / f"layout-{rank}.json").read_text(encoding="utf-8"))
            layout = loaded["layout"]
            stage, lane = divmod(rank, configuration.tp)
            tp_members = list(range(stage * configuration.tp, (stage + 1) * configuration.tp))
            pp_members = list(range(lane, configuration.worker_count, configuration.tp))
            if loaded["tp_members"] != tp_members or loaded["pp_members"] != pp_members or layout["stage"] != stage:
                raise ValueError(f"rank {rank} native groups differ from the requested topology")

            key = json.dumps(layout, sort_keys=True)
            representative = representatives.setdefault(key, rank)
            self.workers[rank] = {
                "representative": representative, "publication_owner": layout["publication_owner"],
                "heads": layout["heads"], "stage": stage, "layers": layout["layers"],
                "tp_members": tp_members, "pp_members": pp_members,
            }
        self.publisher, = (rank for rank, worker in self.workers.items() if worker["publication_owner"])
        if placement == "single_device":
            self.windows = [[rank] for rank in representatives.values()]
        else:
            self.windows = [list(self.workers)]
