"""Select native observation hooks when the worker activates its framework plugin."""

import ctypes
from contextlib import nullcontext

from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable


def install(description):
    runtime = device_runtime()
    device_api = f"torch.{platform_domain().device_type}"
    shard_transport = None

    if description["local_shards"]:
        from prologue.integrations.frameworks.sglang.observation.staging import ShardTransport

        shard_transport = ShardTransport

    # A latency-only worker must start without the observer bridge or a device profiler.
    if description["latency_only"]:
        from prologue.integrations.frameworks.sglang.observation.host_timing import NativeTiming


        if description["host_timing"]:
            timing = NativeTiming(description, host_profile=runtime.host_profile)
        elif "attention_sites" in description:
            from prologue.integrations.frameworks.sglang.observation.model_samples import (
                DecodeAttentionObservation,
            )

            timing = NativeTiming(description, attention=DecodeAttentionObservation)
        else:
            if getattr(ctypes.CDLL(None), "prologue_set_runtime_enabled", None) is not None:
                raise ValueError("latency measurement requires the native observer to be unloaded")
            timing = NativeTiming(description, specializations=runtime.specializations())

        if shard_transport is not None:
            auxiliary = timing.host.auxiliary if timing.host is not None else nullcontext
            record = timing.host.local_exchange if timing.host is not None else None
            shard_transport(description["output"], auxiliary, record).install(description)

        timing.install(device_api)
        return

    from prologue.integrations.frameworks.sglang.observation.observers import SGLangObservation

    HookTable.group(runtime.profiling.hooks()).install()
    markers = runtime.markers(detailed_timing=description["service_reference"])
    SGLangObservation(description, markers).install(device_api, shard_transport)
