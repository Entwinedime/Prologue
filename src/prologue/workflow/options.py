"""Host inputs for a single prediction and a configuration/workload exploration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionOptions:
    output: Path
    binary: Path
    control_image: str
    modeling_image: str
    timeout: float
    execution_trace: bool = False
    execution_inputs: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplorationOptions:
    input: Path
    output: Path | None
    libraries: Path
    binary: Path
    control_image: str
    modeling_image: str
    timeout: float
    details: bool
    prediction_workers: int
    storage_bytes: int | None
    deployment: Path | None
