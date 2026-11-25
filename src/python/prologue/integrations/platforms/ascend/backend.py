"""Ascend capabilities, loaded on first use in the process that owns them."""

from functools import cached_property
from importlib import import_module
from math import prod

from prologue.common.interfaces import PlatformDomain


class Component:
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __set_name__(self, owner, name):
        self.attribute = name

    def __get__(self, instance, owner):
        component = getattr(import_module(self.module, __package__), self.name)
        setattr(instance, self.attribute, component)
        return component


class AscendDeviceRuntime:
    domain = PlatformDomain("ascend", "npu", "torch_npu", "hccl", ("ascend",), "ascend", 200)
    markers = Component(".runtime.profiling", "AscendMarkers")
    profiling = Component(".runtime.profiling", "AscendProfiling")
    host_profile = Component(".runtime.profiling", "AscendHostProfile")
    specializations = Component(".runtime.profiling", "AscendSpecializations")
    export = Component(".runtime.sampling", "AscendExport")
    copy_sampler = Component(".runtime.sampling", "TensorCopySamples")
    scoped_tasks = Component(".analysis.commands", "ScopedTasks")
    tensor_layouts = Component(".analysis.commands", "tensor_layouts")
    observation = Component(".analysis.observation", "AscendObservation")
    expert_environment = {"DEEP_USE_ALLTOALL_MODE": "1", "DEEPEP_HCCL_BUFFSIZE": "1024", "HCCL_BUFFSIZE": "1024"}

    @cached_property
    def api(self):
        import torch
        import torch_npu  # noqa: F401 -- installs the native torch device namespace

        return torch.npu

    @staticmethod
    def kv_token_layout(key, value):
        import torch

        if any(not isinstance(buffer, torch.Tensor) or buffer.ndim != 5 for buffer in (key, value)):
            raise NotImplementedError("KV transfer layout requires contiguous paged MHA buffers")

        # Native page and layer axes precede the per-token head dimensions.
        key_bytes, value_bytes = (prod(buffer.shape[3:]) * buffer.element_size() for buffer in (key, value))
        return str(key.dtype), key_bytes, value_bytes

    @staticmethod
    def load_weight_bridge():
        import torch

        torch.ops.load_library("/opt/prologue/lib/libprologue_ascend_weights.so")

    @staticmethod
    def move_weight(tensor, backing, to_device):
        import torch

        torch.ops.prologue_ascend.move_weight_(tensor, backing, to_device)


class AscendPlatformBackend:
    deployment = Component(".host.deployment", "AscendDeployment")
    host_alignment = Component(".host.services", "AscendHostAlignment")
    collectives = Component(".host.services", "CollectiveServices")
    transport_profile = Component(".host.services", "AscendTransportProfile")
    observed_services = Component(".host.services", "ObservedServices")
    trace = Component(".analysis.trace", "AscendTrace")
    interval = Component(".trace_store", "Interval")
    thread_intervals = Component(".trace_store", "ThreadIntervals")
    recording = Component(".reference.recording", "SdkRecording")
    reference_evidence = Component(".reference.checks", "NativeEffectEvidence")
    stream_effects = Component(".reference.checks", "StreamEffects")
    image_tag = "ubuntu22.04"
    profile_glob = "*_ascend_pt"


device = AscendDeviceRuntime()
platform = AscendPlatformBackend()
