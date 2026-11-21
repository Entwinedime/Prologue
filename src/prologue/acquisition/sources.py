"""Execute a finite source plan using the existing capture and production owners."""

import fcntl
import filecmp
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from prologue.artifacts import write_json
from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.recording import PublishedCapture
from prologue.workflow.limits import AllowanceExhausted, OperationLimit, Usage, retained_bytes
from prologue.acquisition.production import CaptureProduction


class SourceAcquisition:
    def __init__(self, arguments, place, limit=None):
        self.arguments = arguments
        self.place = place
        self.output = arguments.output.resolve()
        plan_root = arguments.plan.resolve(strict=True)
        plan = json.loads((plan_root / "plan.json").read_text())
        deployment = json.loads(arguments.deployment.read_text())
        self.jobs, self.descriptions = {}, {}
        for description in plan["sources"]:
            identity = description["id"]
            if type(identity) is not int or identity < 0 or identity in self.jobs:
                raise ValueError("source plan needs distinct nonnegative source job IDs")
            path = (plan_root / description["source"]["input"]).resolve(strict=True)
            if not path.is_relative_to(plan_root):
                raise ValueError("planned capture input lies outside its plan directory")
            job = json.loads(path.read_text())
            model = Path(job["model"]["path"]).resolve(strict=True)
            if self.output == model or model in self.output.parents:
                raise ValueError("source collection must not write inside a model directory")
            self.jobs[identity] = job
            self.descriptions[identity] = description
        if not self.jobs:
            raise ValueError("source acquisition needs a nonempty plan")

        observations = json.loads(arguments.observations.read_text()) if arguments.observations is not None else {}
        if not isinstance(observations, dict) or observations.keys() - {str(identity) for identity in self.jobs}:
            raise ValueError("observation references name an unplanned source job")
        self.observations = {}
        self.sources = {identity: self.output / "sources" / str(identity) for identity in self.jobs}
        for key, path in observations.items():
            identity = int(key)
            conditions = {"resource_trace": True} if arguments.resource_trace else None
            recording = PublishedCapture(Path(path), self.jobs[identity], conditions=conditions)
            if self.output.is_relative_to(recording.directory):
                raise ValueError("source collection must not write inside a referenced native task")
            self.observations[identity] = recording
            self.sources[identity] = recording.directory

        for value in (arguments.postprocessing_budget_seconds, arguments.timeout):
            if not math.isfinite(value) or value < 1e-9:
                raise ValueError("collection operation and postprocessing times must be finite and positive")
        if (not math.isfinite(arguments.native_budget_seconds) or arguments.native_budget_seconds < 0
                or arguments.attempt_limit < 0 or arguments.storage_bytes <= 0):
            raise ValueError("collection needs nonnegative native allowances and positive storage")
        self.binary = arguments.binary.resolve(strict=True)
        if not self.binary.is_file():
            raise ValueError("source collection requires a compiled executable")

        self.budget = Usage(int(arguments.native_budget_seconds * 1e9), arguments.attempt_limit,
                            int(arguments.postprocessing_budget_seconds * 1e9))
        self.limit = limit if limit is not None else OperationLimit(self.output, arguments.storage_bytes)
        images = {role: subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
                  for role, image in (("source", arguments.source_image), ("modeling", arguments.modeling_image))}
        self.inputs = {"plan": plan, "jobs": {str(key): job for key, job in self.jobs.items()},
                       "deployment": deployment, "images": images, "resource_trace": arguments.resource_trace,
                       "keep_intermediates": arguments.keep_intermediates}
        if self.observations:
            self.inputs["observations"] = {str(key): str(recording.directory) for key, recording in self.observations.items()}

    @staticmethod
    def spending(directory):

        # External observations stay charged to their original owners. Only local attempts enter this total.
        native = (NativeAttempt.usage(path.parent) for path in directory.glob("sources/*/execution.json"))
        processing = (CaptureProduction.usage(path.parent) for path in directory.glob("assets/*/production.json"))
        return Usage.total((*native, *processing))

    def status(self, error):
        write_json(self.output / "status.json", {
            **self.spending(self.output).document(), "retained_bytes": retained_bytes(self.output),
            "limits": {"native_ns": self.budget.native_ns, "postprocessing_ns": self.budget.processing_ns,
                       "native_attempts": self.budget.attempts, "storage_bytes": self.arguments.storage_bytes},
            "error": error,
        }, indent=2)

    @staticmethod
    def release_intermediates(directory):

        # Only published productions owned by this collection may release their template workspace.
        for publication in (directory / "assets").glob("*/result.json"):
            CaptureProduction.release_work(publication.parent)

        status = directory / "status.json"
        if status.is_file():
            report = json.loads(status.read_text())
            report["retained_bytes"] = retained_bytes(directory)
            write_json(status, report, indent=2)

    def prepare(self):
        path = self.output / "input.json"
        executable = self.output / "inputs/prologue"
        if path.exists():
            if json.loads(path.read_text()) != self.inputs or not filecmp.cmp(self.binary, executable, shallow=False):
                raise ValueError("collection inputs, program, images or retention policy changed; use a new directory")
        else:
            if any(path.name != ".lock" for path in self.output.iterdir()):
                raise ValueError("new source collection requires an empty directory")
            executable.parent.mkdir()
            shutil.copyfile(self.binary, executable)
            executable.chmod(0o500)
            write_json(path, self.inputs, indent=2, replace=False)

        # Every source job uses this task's program, even if the external build changes during native capture.
        self.binary = executable

    def placements(self, allow_incomplete):
        launches, unavailable = {}, {}
        for identity, job in self.jobs.items():
            try:
                launch = self.place(job, self.inputs["deployment"])
            except ValueError as failure:
                if not allow_incomplete:
                    raise
                unavailable[str(identity)] = {"status": "native_unavailable", "message": str(failure)}
                continue
            launch.domain.require_capture(launch.serving, self.descriptions[identity])
            if identity in self.observations:
                launch.require_recorded_placement(self.observations[identity].inputs["deployment"])
            launches[identity] = launch
        return launches, unavailable

    def acquire(self, identity, launch):
        source = self.sources[identity]
        assets = self.output / "assets" / str(identity)
        owner = Usage.total((NativeAttempt.usage(self.output / "sources" / str(identity)), CaptureProduction.usage(assets)))
        allowance = self.budget.available_to(owner, self.spending(self.output))
        if allowance.processing_ns <= 0:
            raise AllowanceExhausted("source collection postprocessing allowance exhausted")

        # Explicit observations stay owned by their original task; only missing observations launch an Engine.
        recording = self.observations.get(identity)
        if recording is None:
            launch.capture(SimpleNamespace(
                output=source, timeout=self.arguments.timeout,
                budget_seconds=allowance.native_ns / 1e9, attempt_limit=allowance.attempts,
                source_image=self.inputs["images"]["source"], resource_trace=self.arguments.resource_trace,
            ), self.limit).run()
            recording = PublishedCapture(source, self.jobs[identity])
        CaptureProduction(SimpleNamespace(
            output=assets, binary=self.binary, product="templates",
            budget_seconds=allowance.processing_ns / 1e9, source_image=self.inputs["images"]["source"],
            modeling_image=self.inputs["images"]["modeling"], keep_intermediates=self.arguments.keep_intermediates,
        ), recording, self.place, self.limit).run()
        publication = json.loads((assets / "result.json").read_text())
        libraries = [json.loads((assets / path).read_text()) for path in publication["templates"]]
        launch.domain.require_planned(libraries, self.descriptions[identity], self.inputs["plan"])

    def run(self, *, allow_incomplete=False):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.prepare()
            error = None
            try:
                # Preflight every placement before starting any worker, preserving each ineligible job's reason.
                launches, unavailable = self.placements(allow_incomplete)
                available = []
                for identity, launch in launches.items():
                    try:
                        self.acquire(identity, launch)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, RuntimeError) as failure:
                        if not allow_incomplete:
                            raise
                        self.limit.check()
                        if isinstance(failure, AllowanceExhausted):
                            status = "budget_exhausted"
                        else:
                            recorded = (self.sources[identity] / "result.json").is_file()
                            status = "source_failed" if recorded else "native_unavailable"
                        unavailable[str(identity)] = {"status": status, "message": str(failure)}
                    else:
                        available.append(identity)
                self.limit.check()
                write_json(self.output / "result.json", {
                    "productions": [f"assets/{identity}" for identity in available],
                    "captures": [str(self.sources[identity]) for identity in available],
                    "unavailable": unavailable,
                    "qualification": "partial_sources" if unavailable else "compiled_initial_sources",
                    "limits": self.inputs["plan"]["limits"],
                }, indent=2)
            except BaseException as failure:
                error = f"{type(failure).__name__}: {failure}"
                raise
            finally:
                self.status(error)
