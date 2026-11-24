"""Capture one native job with cumulative accounting and explicit invocation-by-invocation retries."""

import fcntl
import json
import math
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from prologue.artifacts import write_json
from prologue.integrations.runtimes.docker import OwnedContainers
from prologue.workflow.limits import AllowanceExhausted, OperationLimit, Usage


class NativeAttempt:
    @staticmethod
    def image(name):
        return json.loads(subprocess.check_output(["docker", "image", "inspect", name], text=True))[0]

    @staticmethod
    def attempts(directory):
        state = directory / "execution.json"
        return json.loads(state.read_text())["attempts"] if state.is_file() else []

    @classmethod
    def usage(cls, directory):
        return cls.charges(cls.attempts(directory))

    @staticmethod
    def charges(attempts):
        return Usage(sum(row["charged_ns"] for row in attempts), len(attempts))

    def __init__(self, arguments, inputs, options, worker_options, mounts, limit=None):
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
        path = self.output / "execution.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            job = json.loads((self.output / "input.json").read_text())
            if self.state["inputs"] != self.inputs or job != self.inputs["job"]:
                raise ValueError("capture inputs, deployment or image changed; use a new task directory")
            return
        job = self.output / "input.json"
        if any(path.name not in {".lock", job.name} for path in self.output.iterdir()):
            raise ValueError("new capture task requires an empty directory")
        self.state = {"inputs": self.inputs, "attempts": []}
        if not job.exists():
            write_json(job, self.inputs["job"], indent=2, replace=False)
        elif json.loads(job.read_bytes()) != self.inputs["job"]:
            raise ValueError("interrupted capture preparation belongs to a different job")
        self.save()

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
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads((directory / "execution.json").read_bytes())
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
            self.inputs["image"], "--signal=KILL", f"{allowance_ns / 1_000_000_000}s", "/opt/prologue/capture.sh",
            "--input", "/input.json", "--output", "/capture/source",
            *self.worker_options,
        ]
        attempt = {
            "number": number, "container": container, "command": command, "timeout_ns": allowance_ns,
            "charged_ns": allowance_ns, "elapsed_ns": None, "exit_code": None, "status": "running", "error": None,
        }

        # Reserve before launch; the container's own deadline also survives a killed coordinator.
        self.state["attempts"].append(attempt)
        self.save()
        started_ns = time.monotonic_ns()
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
            attempt["elapsed_ns"] = time.monotonic_ns() - started_ns
            attempt["charged_ns"] = attempt["elapsed_ns"]
            attempt["status"] = "exited" if attempt["exit_code"] == 0 else "failed"
            if self.operation_limit.sample() is not None:
                attempt["sampled_collection_peak_bytes"] = self.operation_limit.sampled_peak_bytes
            self.save()

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            capture = json.loads((source / "capture.json").read_text())
            write_json(self.output / "result.json", {
                "source": str(source.relative_to(self.output)), **capture,
                "native_charged_ns": sum(attempt["charged_ns"] for attempt in attempts), "attempts": len(attempts),
            }, indent=2)
        print(f"Created {self.output / 'result.json'}")
