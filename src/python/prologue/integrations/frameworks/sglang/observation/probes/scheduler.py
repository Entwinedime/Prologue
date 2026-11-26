"""Run a scheduler or receiver cut and retain the original boundaries for independent readback."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import read_json, write_json

from ..window import publish_capture
from . import CaptureProbe


class WorkerProbes:
    def __init__(self, revision):
        self.revision = revision

    def run(self, request, output, mode):
        subprocess.run([sys.executable, "-m", "prologue.workers.capture", "capture", "--input", str(request),
                        "--output", str(output), mode], check=True)
        manifest = read_json(output / "capture.json")
        if manifest["framework_revision"] != self.revision or manifest["engine_starts"]:
            raise ValueError("local probes must execute native worker cuts on the same framework")
        return manifest

    def pair(self, request, output):
        modes = {"structure": "--local-worker", "timing": "--local-host"}

        # Separate processes keep structure instrumentation outside the CPU service measurement.
        starts = sum(self.run(request, output / name, mode)["model_runner_starts"] for name, mode in modes.items())
        publish_capture(output, self.revision, starts, probes={name: name for name in modes})

    def calibration(self, job, output):
        cuts = job.pop("calibration")
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output) as temporary:
            request = Path(temporary) / "job.json"
            write_json(request, job)
            primary = self.run(request, output, "--local-worker")
            paths = {}

            for name, batch in cuts.items():
                paths[name] = f"calibration/{name}"
                write_json(request, job | {"batch": batch})
                measured = self.run(request, output / paths[name], "--local-host-pair")
                primary["model_runner_starts"] += measured["model_runner_starts"]

        write_json(output / "capture.json", primary | {"calibration": paths})


def local_worker(invocation):
    job = invocation.job
    if "calibration" in job:
        WorkerProbes(invocation.revision).calibration(job, invocation.output)
        return

    invocation.enable_runtime_queues()
    from ..scheduler_samples import SchedulerRunner

    SchedulerRunner(invocation.job, invocation.output).run(invocation.revision, host_timing=False)


def host_timing(invocation):
    from ..scheduler_samples import SchedulerRunner

    SchedulerRunner(invocation.job, invocation.output).run(invocation.revision, host_timing=True)


def prebuilt(invocation):
    from prologue.integrations.frameworks.sglang.observation.receiver_samples import PrebuiltRunner

    PrebuiltRunner(invocation.job, invocation.output).run(invocation.revision)


probes = (
    CaptureProbe("local-worker", "read the initialized scheduler, model and pools", local_worker),
    CaptureProbe(
        "local-host", "measure local scheduler work with the independent CPU probe", host_timing, runtime_queues=True
    ),
    CaptureProbe("local-prebuilt", "observe the receiver assembling a native PD batch", prebuilt, runtime_queues=True),
)
