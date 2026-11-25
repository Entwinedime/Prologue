"""Host inputs for measuring and qualifying native evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureOptions:
    output: Path
    timeout: float
    budget_seconds: float
    attempt_limit: int
    source_image: str
    resource_trace: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class QualificationOptions:
    binary: Path | None
    modeling_image: str
    timeout: float
    structure_session: Sequence[Path] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalAcquisitionOptions(CaptureOptions):
    binary: Path | None
    modeling_image: str
    structure_session: Sequence[Path] = ()
    storage_bytes: int | None = None
    card_budget_ns: int | None = None
    preparation_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductionOptions:
    output: Path
    binary: Path
    source_image: str
    modeling_image: str
    product: str
    budget_seconds: float
    keep_intermediates: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class FitCostsOptions:
    output: Path
    binary: Path | None
    modeling_image: str
    timeout: float
    samples: Sequence[Path]
    gap: Sequence[Path]
    demand: Sequence[tuple[Path, Path]]
    check: Sequence[Path]


@dataclass(frozen=True, slots=True, kw_only=True)
class CostAcquisitionOptions:
    output: Path
    binary: Path
    source_image: str
    modeling_image: str
    timeout: float
    storage_bytes: int
    native_budget_seconds: float
    postprocessing_budget_seconds: float
    attempt_limit: int
    keep_intermediates: bool
    production: Sequence[Path]
    samples: Sequence[Path]
    gap: Sequence[Path]
    demand: Sequence[tuple[Path, Path]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplenishmentOptions:
    deployment: Path
    source_image: str
    timeout: float
    card_budget_seconds: float | None
    postprocessing_budget_seconds: float | None
    attempt_limit: int | None
    storage_bytes: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class HostSampleOptions:
    detailed: Path
    host: Path
    facts: Path
    trace: Path
    graph: Path
    work_identities: Path
    output: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceSampleOptions:
    production: Path
    projection: Path | None
    rank: Sequence[int]
    work_identities: Sequence[Path]
    output: Sequence[Path]
    transport_only: bool = False
