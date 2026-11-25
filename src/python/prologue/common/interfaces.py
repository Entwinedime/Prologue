"""The platform contract shared by device capture, host analysis and CPU-only control."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PlatformDomain:
    name: str
    device_type: str
    runtime_domain_key: str
    collective_backend: str
    strided_attention: tuple[str, ...]
    pd_transfer: str


class DeviceRuntime(Protocol):
    domain: PlatformDomain
    api: Any
    markers: type
    profiling: type
    host_profile: type
    specializations: type
    export: type
    copy_sampler: type
    scoped_tasks: type
    observation: type
    tensor_layouts: Callable
    expert_environment: dict[str, str]

    def load_weight_bridge(self) -> None: ...
    def move_weight(self, tensor, backing, to_device: bool) -> None: ...


class Deployment(Protocol):
    driver_mounts: tuple[str, ...]

    @staticmethod
    def source_docker_runtime() -> str: ...

    @staticmethod
    def capture_entrypoint() -> str: ...


class PlatformBackend(Protocol):
    deployment: type[Deployment]
    host_alignment: type
    collectives: type
    transport_profile: type
    thread_service: type
    observed_services: type
    trace: type
    calibration: type
    interval: type
    thread_intervals: type
    recording: type
    reference_evidence: type
    stream_effects: type
    profile_glob: str
