"""Registered structural facts that can be checked after composing the verified case."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ...core.native_case import NativeCase
from ..trace_events import PredictionEvents


@dataclass(frozen=True, slots=True)
class ActionCheck:
    """An independent check over qualified case data and decoded prediction records."""

    name: str
    findings: Callable[[NativeCase, PredictionEvents], Iterable[tuple[str, dict, dict]]]
    evidence_gaps: frozenset[str] = frozenset()
