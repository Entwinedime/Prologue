"""Ascend capabilities, loaded on first use in the process that owns them."""

from functools import cached_property
from importlib import import_module

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
    domain = PlatformDomain("ascend", "npu", "torch_npu", "hccl", ("ascend",), "ascend")
    markers = Component(".runtime.markers", "AscendMarkers")
    profiling = Component(".runtime.markers", "AscendProfiling")
    host_profile = Component(".runtime.host_profile", "AscendHostProfile")
    specializations = Component(".runtime.host_profile", "AscendSpecializations")
    export = Component(".runtime.export", "AscendExport")
    copy_sampler = Component(".runtime.copy_sampling", "TensorCopySamples")
    scoped_tasks = Component(".analysis.scoped_tasks", "ScopedTasks")
    tensor_layouts = Component(".analysis.scoped_tasks", "tensor_layouts")
    observation = Component(".analysis.observation", "AscendObservation")
    expert_environment = {"DEEP_USE_ALLTOALL_MODE": "1", "DEEPEP_HCCL_BUFFSIZE": "1024", "HCCL_BUFFSIZE": "1024"}

    @cached_property
    def api(self):
        import torch
        import torch_npu  # noqa: F401 -- installs the native torch device namespace

        return torch.npu

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
    host_alignment = Component(".analysis.host_alignment", "AscendHostAlignment")
    collectives = Component(".host.collectives", "CollectiveServices")
    transport_profile = Component(".host.transport_profile", "AscendTransportProfile")
    thread_service = Component(".analysis.host_alignment", "ThreadService")
    observed_services = Component(".analysis.services", "ObservedServices")
    trace = Component(".analysis.trace", "AscendTrace")
    calibration = Component(".analysis.clock", "DeviceCalibration")
    interval = Component(".analysis.trace", "Interval")
    thread_intervals = Component(".analysis.trace", "ThreadIntervals")
    recording = Component(".reference.recording", "SdkRecording")
    reference_evidence = Component(".reference.reference", "NativeEffectEvidence")
    stream_effects = Component(".reference.stream_effects", "StreamEffects")
    profile_glob = "*_ascend_pt"


device = AscendDeviceRuntime()
platform = AscendPlatformBackend()
