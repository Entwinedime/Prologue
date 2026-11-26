"""Capture one native job with cumulative accounting and explicit invocation-by-invocation retries."""

import json
import math
import subprocess
from pathlib import Path
from uuid import uuid4

from prologue.common.interfaces import Deployment
from prologue.common.io import read_json, write_json
from prologue.host.acquisition.options import CaptureOptions
from prologue.host.runtime.docker import OwnedContainers
from prologue.host.runtime.limits import AllowanceExhausted, OperationLimit, Usage
from prologue.host.runtime.workspace import TaskWorkspace, file_lock


class NativeAttempt:
    """Own a single container execution and its durable budget receipt. Later requalification can read the
    recording without acquiring more device time.
    """

    @staticmethod
    def attempts(directory):
        state = directory / "execution.json"
        return read_json(state)["attempts"] if state.is_file() else []

    @classmethod
    def usage(cls, directory):
        return cls.charges(cls.attempts(directory))

    @staticmethod
    def charges(attempts):
        return Usage(sum(row["charged_ns"] for row in attempts), len(attempts))

    def __init__(
        self,
        arguments: CaptureOptions,
        inputs,
        options,
        worker_options,
        mounts,
        runtime: type[Deployment],
        limit=None,
    ):
        self.entrypoint = runtime.capture_entrypoint()
        self.arguments, self.inputs = arguments, inputs
        self.options, self.worker_options, self.mounts = options, worker_options, mounts
        self.output = arguments.output.resolve()
        self.operation_limit = limit if limit is not None else OperationLimit()

        if not math.isfinite(arguments.timeout) or arguments.timeout < 1e-9:
            raise ValueError("native timeout must be finite and at least one nanosecond")
        if not math.isfinite(arguments.budget_seconds) or arguments.budget_seconds < 0 or arguments.attempt_limit < 0:
            raise ValueError("native cumulative time and attempt allowances must be nonnegative")

        self.limit_ns = int(arguments.budget_seconds * 1_000_000_000)
        self.timeout_ns = int(arguments.timeout * 1_000_000_000)

    def save(self) -> None:
        write_json(self.output / "execution.json", self.state, indent=2)

    def prepare(self) -> None:
        with TaskWorkspace(self.output).initialize("execution.json") as draft:
            if draft is not None:
                self.state = {"inputs": self.inputs, "attempts": []}
                write_json(draft / "input.json", self.inputs["job"], indent=2, replace=False)
                write_json(draft / "execution.json", self.state, indent=2)
                return
        self.state = read_json(self.output / "execution.json")
        if self.state["inputs"] != self.inputs or read_json(self.output / "input.json") != self.inputs["job"]:
            raise ValueError("capture inputs, deployment or image changed; use a new task directory")

    @staticmethod
    def recover(directory, state):
        attempt = state["attempts"][-1]
        owned = OwnedContainers([attempt["container"]])
        if attempt["exit_code"] is None and owned.remaining():
            status = json.loads(subprocess.check_output([
                "docker", "inspect", attempt["container"], "--format", "{{json .State}}",
            ], text=True))

            if status["Status"] == "exited":
                attempt["exit_code"] = status["ExitCode"]

        # A retained container supplies exit evidence after client interruption; a raw manifest alone does not.
        owned.close()
        attempt["status"] = "exited" if attempt["exit_code"] == 0 else "failed"
        attempt["error"] = (None if attempt["status"] == "exited"
                            else "interrupted native attempt; reserved time remains charged")
        write_json(directory / "execution.json", state, indent=2)

    @classmethod
    def recover_pending(cls, directory):
        if not (attempts := cls.attempts(directory)) or attempts[-1]["status"] != "running":
            return
        with file_lock(directory / ".lock"):
            state = read_json(directory / "execution.json")
            if state["attempts"][-1]["status"] == "running":
                cls.recover(directory, state)

    def capture(self, allowance_ns: int) -> None:
        number = len(self.state["attempts"]) + 1
        output = self.output / "attempts" / str(number)
        container = f"prologue-capture-{uuid4().hex}"
        command = [
            "docker", "run", "--init", "--name", container, *self.options,
            "--workdir", str(Path.cwd()), "--entrypoint", "/usr/bin/timeout",
            *self.mounts,
            "--mount", f"type=bind,src={self.output / 'input.json'},dst=/input.json,readonly",
            "--mount", f"type=bind,src={output},dst=/capture",
            self.inputs["image"], "--signal=KILL", f"{allowance_ns / 1_000_000_000}s", self.entrypoint,
            "--input", "/input.json", "--output", "/capture/source",
            *self.worker_options,
        ]
        attempt = {
            "number": number, "container": container, "command": command, "timeout_ns": allowance_ns,
            "charged_ns": 0, "elapsed_ns": None, "exit_code": None, "status": "running", "error": None,
        }

        # Reserve before launch; the container's own deadline also survives a killed coordinator.
        self.state["attempts"].append(attempt)
        with TaskWorkspace.reserve(attempt, "charged_ns", allowance_ns, self.save, elapsed_field="elapsed_ns",
                                   settled=lambda: attempt["status"] != "running"):
            try:
                output.mkdir(parents=True)
                with (output / "capture.log").open("x") as log:
                    attempt["exit_code"] = self.operation_limit.run(command, log, allowance_ns / 1_000_000_000)

                self.save()
                if attempt["exit_code"]:
                    raise subprocess.CalledProcessError(attempt["exit_code"], command)
            except BaseException as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                OwnedContainers([container]).close()
                attempt["status"] = "exited" if attempt["exit_code"] == 0 else "failed"
                if self.operation_limit.sample() is not None:
                    attempt["sampled_collection_peak_bytes"] = self.operation_limit.sampled_peak_bytes

    def run(self) -> None:
        with file_lock(self.output / ".lock"):
            self.prepare()
            attempts = self.state["attempts"]
            if attempts and attempts[-1]["status"] == "running":
                self.recover(self.output, self.state)

            spent_ns = sum(attempt["charged_ns"] for attempt in attempts)
            if not attempts or attempts[-1]["status"] != "exited":
                if len(attempts) >= self.arguments.attempt_limit or spent_ns >= self.limit_ns:
                    raise AllowanceExhausted("cumulative native time or attempt allowance exhausted")
                self.operation_limit.check()
                self.capture(min(self.timeout_ns, self.limit_ns - spent_ns))

            # Publish from the successful attempt even if the coordinator stopped after the worker exited.
            source = self.output / "attempts" / str(attempts[-1]["number"]) / "source"
            capture = read_json(source / "capture.json")
            write_json(self.output / "result.json", {
                "source": str(source.relative_to(self.output)), **capture,
                "native_charged_ns": sum(attempt["charged_ns"] for attempt in attempts), "attempts": len(attempts),
            }, indent=2)

        print(f"Created {self.output / 'result.json'}")
