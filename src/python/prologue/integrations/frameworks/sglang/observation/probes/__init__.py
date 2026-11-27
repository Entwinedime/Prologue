"""Register native probes without loading model or device libraries during CLI parsing."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProbeInput:
    """One native invocation in a process owned by the capture worker."""

    source: Path
    output: Path
    job: dict
    revision: str

    def enable_runtime_queues(self):

        # Native classes may open their sinks as soon as they are imported.
        runtime = self.output / "runtime"
        runtime.mkdir(parents=True)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(runtime)


@dataclass(frozen=True, slots=True)
class CaptureProbe:
    """The worker switch and process prerequisites of one installed native cut."""

    name: str
    description: str
    execute: Callable[[ProbeInput], None]
    runtime_queues: bool = False

    def run(self, invocation):
        if self.runtime_queues:
            invocation.enable_runtime_queues()

        self.execute(invocation)
