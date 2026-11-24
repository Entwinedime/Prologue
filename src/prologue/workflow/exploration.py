"""Predict each requested configuration and workload against the same immutable release."""

import fcntl
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace

from prologue.artifacts import write_json
from prologue.libraries import LibraryRelease
from prologue.workflow.limits import OperationLimit, Usage, retained_bytes
from prologue.workflow.report import ExplorationReport
from prologue.workflow.targets import ExplorationTarget


class ExplorationTask:
    def __init__(self, args, predictor):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.root = (args.output or Path("data") / f"{stamp}_exploration").resolve()
        self.release = LibraryRelease(args.libraries)
        self.predictor = predictor
        self.binary = args.binary.resolve(strict=True)
        self.source = json.loads(args.input.read_bytes())
        if self.source.keys() != {"framework", "model", "configurations", "workloads"} or self.source["framework"] != "sglang":
            raise ValueError("exploration requires an SGLang model, configurations and workloads")
        if self.root.is_relative_to(self.release.directory) or self.release.directory.is_relative_to(self.root):
            raise ValueError("exploration output and published libraries must be separate")
        if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.storage_bytes is not None and args.storage_bytes <= 0):
            raise ValueError("prediction timeout and optional storage allowance must be positive")

        self.configurations = self.named("configurations")
        self.workloads = self.named("workloads")
        if any(row.keys() != {"id", "instances"} for row in self.configurations):
            raise ValueError("each configuration needs an id and serving instances")
        self.limit = OperationLimit(self.root, args.storage_bytes) if args.storage_bytes is not None else OperationLimit()
        self.runtime = {name: subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", getattr(args, name)], text=True,
        ).strip() for name in ("control_image", "modeling_image")}
        self.run_options = {"timeout": args.timeout, "execution_trace": args.details, **self.runtime}

    def named(self, key):
        rows = self.source[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} must be a nonempty array")
        names = [row["id"] for row in rows]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"{key} require unique nonempty IDs")
        return rows

    def prepare(self):
        directory = self.root / "inputs"
        manifest = directory / "task.json"
        executable = directory / "prologue"
        identity = {"input": self.source, "libraries": str(self.release.directory), "release_id": self.release.identity, "images": self.runtime,
                    "binary": hashlib.sha256(self.binary.read_bytes()).hexdigest(), "details": self.run_options["execution_trace"]}
        if manifest.exists():
            if json.loads(manifest.read_bytes()) != identity or hashlib.sha256(executable.read_bytes()).hexdigest() != identity["binary"]:
                raise ValueError("exploration input, release or runtime changed; use a new output directory")
        else:
            staging = self.root / ".prepare-inputs"
            if any(path.name not in {".lock", staging.name} for path in self.root.iterdir()):
                raise ValueError("new exploration needs an empty output directory")

            # An interruption before the rename leaves only the private staging area to finish on resume.
            staging.mkdir(exist_ok=True)
            snapshot = staging / executable.name
            shutil.copyfile(self.binary, snapshot)
            snapshot.chmod(0o700)
            (staging / manifest.name).write_text(json.dumps(identity, indent=2) + "\n")
            staging.rename(directory)
        self.run_options["binary"] = executable

    def explore(self):
        targets = []
        for index, (configuration, workload) in enumerate(product(self.configurations, self.workloads)):
            source = {"model": self.source["model"], "instances": configuration["instances"], "workload": workload}
            target = ExplorationTarget(self.root, index, source, configuration["id"])
            options = SimpleNamespace(**self.run_options, execution_inputs=None)
            targets.append(target.predict(self.predictor, options, self.release, self.limit))

        usage = {**Usage().document(), "retained_bytes": retained_bytes(self.root)}
        return ExplorationReport(targets, self.release.directory, usage)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.prepare()
            result = self.root / "result.json"
            if not result.is_file():
                report = self.explore()
                (self.root / "summary.md").write_text(report.markdown())
                write_json(result, report.document(), indent=2, replace=False)
        print(self.root / "summary.md")
        return json.loads(result.read_bytes())["status"] == "complete"
