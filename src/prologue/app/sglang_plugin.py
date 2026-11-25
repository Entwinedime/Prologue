"""Install collection hooks for the framework and platform selected by the source image."""

import ctypes
import json
import os
from contextlib import nullcontext
from pathlib import Path


def install() -> None:
    session = os.environ.get("PROLOGUE_GRAPH_SESSION")
    if session is None:
        return
    description = json.loads(Path(session).read_text())
    device_api = "torch.npu"
    shard_transport = None
    if description["local_shards"]:
        from prologue.integrations.frameworks.sglang.observation.local.shard_transport import ShardTransport

        shard_transport = ShardTransport

    # A latency-only worker must start without the observer bridge or a device profiler.
    if description["latency_only"]:
        from prologue.integrations.frameworks.sglang.observation.timing import NativeTiming
        from prologue.integrations.platforms.ascend.host_profile import AscendHostProfile, AscendSpecializations

        if description["host_timing"]:
            timing = NativeTiming(description, host_profile=AscendHostProfile)
        elif "attention_sites" in description:
            from prologue.integrations.frameworks.sglang.observation.local.attention_observation import DecodeAttentionObservation

            timing = NativeTiming(description, attention=DecodeAttentionObservation)
        else:
            if getattr(ctypes.CDLL(None), "prologue_set_runtime_enabled", None) is not None:
                raise ValueError("latency measurement requires the native observer to be unloaded")
            timing = NativeTiming(description, specializations=AscendSpecializations())
        if shard_transport is not None:
            auxiliary = timing.host.auxiliary if timing.host is not None else nullcontext
            record = timing.host.local_exchange if timing.host is not None else None
            shard_transport(description["output"], auxiliary, record).install(description)
        timing.install(device_api)
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from prologue.integrations.frameworks.sglang.observation.hooks import SGLangObservation
    from prologue.integrations.platforms.ascend.markers import AscendMarkers
    from prologue.integrations.platforms.ascend.profiling import AscendProfiling

    for target, hook in AscendProfiling.hooks().items():
        HookRegistry.register(target, hook, HookType.AROUND)
    markers = AscendMarkers(detailed_timing=description["service_reference"])
    SGLangObservation(description, markers).install(device_api, shard_transport)
