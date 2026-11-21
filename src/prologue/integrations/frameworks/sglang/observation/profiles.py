"""Publish native profiles and the CPU protocol facts needed to connect participants."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.observation.window import ObservationWindow


class NativeProfiles:
    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.windows_by_round = None
        self.round = 0
        from prologue.integrations.runtimes.pytorch.protocol import CpuProtocol

        capture = observation.capture
        self.protocol = CpuProtocol(
            observation.description["native_participants"], observation.calls.communicating,
            capture.current, capture.runtime.service_scope,
        )

    @staticmethod
    def windows(directory, observed, single_device, instances):
        if not single_device:
            return [list(observed)]
        if instances > 1:

            # Both PD pools execute every request. Exactly one role across both pools owns the device profiler.
            plans = [directory / "instances" / str(index) / "role-plan.json" for index in range(instances)]
            while not all(path.is_file() for path in plans):
                time.sleep(0.05)
            observed = [rank for path in plans for rank in json.loads(path.read_bytes())["observed_roles"]]
        return [[rank] for rank in observed]

    @staticmethod
    def protocol_directory(directory, selected, single_device):
        root = directory / "protocol"
        return root / f"worker-{selected[0]}" if single_device else root

    @staticmethod
    def wait(directory: Path, ranks, participants, protocol_directory) -> None:
        published = [directory / "native" / f"rank-{rank}" for rank in ranks]
        protocols = [protocol_directory / f"rank-{rank}.json" for rank in participants]

        # A new observer can take the device only after the previous profiler closed and every participant published.
        while not all(path.is_dir() for path in published) or not all(path.is_file() for path in protocols):
            time.sleep(0.05)

    def request(self, original: Any, manager: Any, request: Any) -> Any:
        from sglang.srt.managers.io_struct import ProfileReqOutput, ProfileReqType

        owner = self.observation
        if request.req_type == ProfileReqType.START_PROFILE:

            # All instances have drained warmup and published their real pool layouts before this request.
            if self.windows_by_round is None:
                self.windows_by_round = self.windows(owner.directory, owner.plan_roles(), owner.description["local_shards"],
                                                     owner.description["instance_count"])
            else:

                # The preceding request drained and flushed its KV. This replay owns fresh ingress and group receipts.
                owner.window = ObservationWindow(owner.directory, owner.description)
            selected = self.windows_by_round[self.round]
            owner.selected = owner.parallel["rank"] in selected
            self.protocol_root = self.protocol_directory(owner.directory, selected, owner.description["local_shards"])
            if owner.description["local_shards"]:

                # A later replay can reuse a registration with a different request slot. Keep this window's peer facts together.
                owner.kv_pools = self.protocol_root / "kv-pools"
        if owner.selected:
            result = original(manager, request)
        else:
            if request.req_type == ProfileReqType.STOP_PROFILE:
                import torch.distributed as distributed

                # Non-exporting participants still join the native profiler's stop consensus.
                distributed.barrier(manager.dp_tp_cpu_group)
            result = ProfileReqOutput(success=True, message="Worker has no assigned observation")
        if request.req_type == ProfileReqType.START_PROFILE and result.success:
            self.protocol.begin()
            owner.window.prepared_phase = "profiler"
        elif request.req_type == ProfileReqType.STOP_PROFILE and result.success:

            # Protocol facts cover every participant, including workers represented by another device profile.
            write_json(self.protocol_root / f"rank-{owner.parallel['rank']}.json", self.protocol.document(), replace=False)
            self.round += 1
        return result

    def start(self, original: Any, manager: Any, output_dir: str, *args: Any, **kwargs: Any) -> Any:
        rank = self.observation.parallel["rank"]
        self.destination = Path(output_dir) / f"rank-{rank}"
        return original(manager, str(self.destination.with_name(f".rank-{rank}")), *args, **kwargs)

    def stop(self, original: Any, manager: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(manager, *args, **kwargs)
        if result.success:
            manager.torch_profiler_output_dir.rename(self.destination)
            if self.observation.capture.closed:
                self.observation.capture.runtime.release()
        return result

    def install(self) -> None:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        for target, hook in self.protocol.hooks():
            HookRegistry.register(target, hook, HookType.AROUND)

        manager = "sglang.srt.managers.scheduler_components.profiler_manager.SchedulerProfilerManager."
        targets = {
            manager + "_profile": self.request,
            manager + "_init_profile": self.start,
            manager + "_stop_profile": self.stop,
        }
        for target, hook in targets.items():
            HookRegistry.register(target, hook, HookType.AROUND)
