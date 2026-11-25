"""Native operations selected by the application and consumed by host acquisition."""

from typing import Protocol

from prologue.common.interfaces import Deployment
from prologue.host.acquisition.options import DeviceSampleOptions, HostSampleOptions


class ProductionAdapter(Protocol):
    @staticmethod
    def platform_runtime() -> type[Deployment]: ...

    @staticmethod
    def image_revision(image: dict) -> str: ...

    @staticmethod
    def sampling_input(job: dict) -> dict | None: ...

    @staticmethod
    def export_profile(rank: int, output: str) -> list[str]: ...

    @staticmethod
    def interpret(rank: int, trace: str, output: str, *, evidence: bool) -> list[str]: ...

    @staticmethod
    def sampling_cuts(rank: int, identities: str, output: str) -> list[str]: ...


class CostSampleAdapter(Protocol):
    @staticmethod
    def host_samples(options: HostSampleOptions) -> None: ...

    @staticmethod
    def device_samples(options: DeviceSampleOptions) -> None: ...
