"""Run one bounded native attempt and keep its charge across restarts."""

import fcntl
import json
import subprocess
from pathlib import Path
from time import monotonic_ns


class Acquisition:
    def __init__(self, output, specification):
        self.output = output.resolve()
        self.specification = specification
        self.plan = self.output / "plan.json"
        self.attempts = self.output / "attempts"

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write(path, document):
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)

    @staticmethod
    def retained(directory):
        record = directory / "attempt.json"
        return sum(path.stat().st_size for path in directory.rglob("*") if path != record and path.is_file())

    def recover(self):
        attempts = []
        for directory in sorted(self.attempts.iterdir()):
            path = directory / "attempt.json"
            if not path.exists():
                # Reservation is persisted before launching any native process.
                continue
            receipt = self.read(path)
            if receipt["status"] == "running":
                receipt["charged_card_ns"] = receipt["reserved_card_ns"]
                receipt["charge_basis"] = "reservation"
                self.finish(directory, receipt)
            attempts.append(receipt)
        return attempts

    def finish(self, directory, receipt):
        record = directory / "attempt.json"
        native = directory / "native.json"
        if native.exists() and receipt.get("exit", 0) == 0:
            receipt["native"] = self.read(native)
            receipt["status"] = receipt["native"]["status"]
        else:
            receipt["status"] = "failed" if "exit" in receipt else "interrupted"
        receipt["retained_bytes"] = self.retained(directory)
        self.write(record, receipt)

    def run(self, command, *, max_attempts, max_card_ns, max_bytes):
        if min(max_attempts, max_card_ns, max_bytes) < 0:
            raise ValueError("acquisition limits must be nonnegative")
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".acquisition.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.plan.exists() and self.read(self.plan) != self.specification:
                raise ValueError("the output directory is pinned to a different sampling plan")
            if not self.plan.exists():
                self.write(self.plan, self.specification)
            self.attempts.mkdir(exist_ok=True)

            attempts = self.recover()
            charged = sum(receipt["charged_card_ns"] for receipt in attempts)
            used = sum(receipt["retained_bytes"] for receipt in attempts)
            if used > max_bytes:
                return self.report("output_budget_exceeded", attempts)
            if attempts and attempts[-1]["status"] in ("observed", "untriggered"):
                return self.report(attempts[-1]["status"], attempts, reused=True)
            remaining = max_card_ns - charged
            if remaining < 1_000_000 or len(attempts) >= max_attempts or used >= max_bytes:
                return self.report("budget_exhausted", attempts)

            # The child timeout retains the lock if the controlling process exits before receiving its result.
            directory = self.attempts / f"{len(attempts) + 1:04}"
            directory.mkdir(exist_ok=True)
            receipt = {"id": directory.name, "status": "running", "reserved_card_ns": remaining}
            self.write(directory / "attempt.json", receipt)
            timeout = f"{remaining // 1_000_000 / 1000:.3f}s"
            started = monotonic_ns()
            with (directory / "native.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(["timeout", "--signal=KILL", timeout, *command, self.plan, directory],
                                         stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            receipt.update(exit=process.returncode, charged_card_ns=monotonic_ns() - started, charge_basis="elapsed")
            self.finish(directory, receipt)
            attempts.append(receipt)
            status = "output_budget_exceeded" if used + receipt["retained_bytes"] > max_bytes else receipt["status"]
            return self.report(status, attempts)

    def report(self, status, attempts, reused=False):
        return {"status": status, "reused": reused, "attempts": attempts,
                "charged_card_ns": sum(row["charged_card_ns"] for row in attempts)}


class ProcessingBudget:
    def __init__(self, path, seconds):
        self.path = path
        self.limit = seconds * 1_000_000_000

    def __enter__(self):
        self.started = monotonic_ns()
        self.boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        self.previous = 0
        if self.path.exists():
            record = Acquisition.read(self.path)
            self.previous = record["charged_ns"]
            if "started_ns" in record:
                # A crash cannot restore time reserved for an action in progress.
                self.previous = (self.previous + self.started - record["started_ns"]
                                 if self.boot == record["boot"] else self.limit)
        Acquisition.write(self.path, {"charged_ns": self.previous, "started_ns": self.started, "boot": self.boot})
        return self

    def __exit__(self, *exception):
        Acquisition.write(self.path, {"charged_ns": self.charged_ns})

    @property
    def charged_ns(self):
        return self.previous + monotonic_ns() - self.started

    def remaining(self):
        remaining = (self.limit - self.charged_ns) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("the exploration processing budget is exhausted")
        return remaining


def acquisition_usage(directory, *, exclude=None):
    totals = {"attempts": 0, "card_ns": 0, "retained_bytes": 0, "engine_starts": 0}
    for path in directory.glob("*/attempts/*/attempt.json"):
        if path.parents[2] == exclude:
            continue
        row = Acquisition.read(path)
        totals["attempts"] += 1
        totals["card_ns"] += row.get("charged_card_ns", row["reserved_card_ns"])
        retained = row["retained_bytes"] if "retained_bytes" in row else Acquisition.retained(path.parent)
        totals["retained_bytes"] += retained
        engines = row.get("native", {}).get("engine_starts")
        if engines is None:
            # An incomplete native attempt cannot prove that no engine started.
            totals["engine_starts"] = None
        elif totals["engine_starts"] is not None:
            totals["engine_starts"] += engines
    return totals
