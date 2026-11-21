"""Prepare assets once and predict a configuration/workload set."""

import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace

from prologue.acquisition.costs import CostAcquisition
from prologue.acquisition.fitting import TransportSamples
from prologue.acquisition.planning import PlanningProcess
from prologue.acquisition.publication import SourceProductions
from prologue.acquisition.sources import SourceAcquisition
from prologue.artifacts import write_json
from prologue.performance.requirements import CostRequirements, WorkPoint
from prologue.workflow.limits import AllowanceExhausted, OperationLimit, Usage, retained_bytes
from prologue.workflow.report import ExplorationReport
from prologue.workflow.targets import ExplorationAsset, ExplorationTarget


def read(path):
    return json.loads(path.read_text())


class ExplorationTask:
    def __init__(self, args, *, planner, place, reuse_samples, predictor):
        self.args = args
        self.place = place
        self.reuse_samples = reuse_samples
        self.predictor = predictor
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.root = (args.output or Path("data") / f"{stamp}_exploration").resolve()
        self.binary = args.binary.resolve(strict=True)
        self.runtime = {}
        for role in ("source_image", "modeling_image", "control_image"):
            self.runtime[role] = subprocess.check_output(
                ["docker", "image", "inspect", getattr(args, role), "--format", "{{.Id}}"], text=True,
            ).strip()
        self.source = read(args.input)
        model = Path(self.source["model"]["path"]).resolve(strict=True)
        self.source["model"] = {**self.source["model"], "path": str(model)}
        deployment = read(args.deployment)
        self.planner = planner(self.source, deployment)
        self.reuse = args.reuse.resolve(strict=True) if args.reuse else None
        if self.root.is_relative_to(model) or (self.reuse and self.root.is_relative_to(self.reuse)):
            raise ValueError("exploration must own an output outside the model and prior task")

        seconds = (args.timeout, args.native_budget_seconds, args.postprocessing_budget_seconds)
        if any(not math.isfinite(value) or value < 0 for value in seconds) or args.timeout <= 0:
            raise ValueError("exploration needs a positive timeout and nonnegative finite time budgets")
        if args.storage_bytes <= 0 or args.attempt_limit < 0:
            raise ValueError("exploration needs nonnegative attempts and positive storage")
        self.budget = Usage(int(args.native_budget_seconds * 1e9), args.attempt_limit,
                            int(args.postprocessing_budget_seconds * 1e9))
        self.limit = OperationLimit(self.root, args.storage_bytes)
        self.environment = {
            "framework": self.source["framework"], "model": self.source["model"], "deployment": deployment,
            "model_config": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "binary": hashlib.sha256(self.binary.read_bytes()).hexdigest(), "images": self.runtime,
        }
        self.inputs = {"source": self.source, "environment": self.environment, "details": args.details,
                       "reuse": str(self.reuse) if self.reuse else None}
        self.transport_samples = TransportSamples(args.transport_samples)
        if self.transport_samples.selection:
            self.inputs["transport_samples"] = self.transport_samples.selection
        self.training = self.root / "collection"
        self.cost_task = self.root / "assets"
        self.reuse_status = None

    def prepare(self):
        directory = self.root / "inputs"
        manifest = directory / "task.json"
        executable = directory / "prologue"
        if manifest.is_file():
            binary = hashlib.sha256(executable.read_bytes()).hexdigest()
            if self.inputs != read(manifest) or binary != self.environment["binary"]:
                raise ValueError("exploration input, runtime or executable changed; use a new task")
        else:
            if any(path.name != ".lock" for path in self.root.iterdir()):
                raise ValueError("new exploration needs an empty output directory")
            directory.mkdir()
            shutil.copyfile(self.binary, executable)
            executable.chmod(0o500)
            write_json(directory / "deployment.json", self.environment["deployment"], indent=2)
            write_json(manifest, self.inputs, indent=2, replace=False)
        self.binary = executable

    def spending(self):
        charges = {self.training: SourceAcquisition.spending(self.training)}
        for state in self.cost_task.glob("*/execution.json"):
            charges[state.parent] = CostAcquisition.spending(state.parent)
        return charges

    def allowance(self, owner):
        charges = self.spending()
        available = self.budget.available_to(charges.get(owner, Usage()), Usage.total(charges.values()))
        return {
            "native_budget_seconds": available.native_ns / 1e9, "attempt_limit": available.attempts,
            "postprocessing_budget_seconds": available.processing_ns / 1e9, "storage_bytes": self.args.storage_bytes,
            "timeout": self.args.timeout, "binary": self.binary, "source_image": self.runtime["source_image"],
            "modeling_image": self.runtime["modeling_image"],
        }

    def previous(self):
        if self.reuse is None:
            return []
        prior = read(self.reuse / "inputs/task.json")
        if prior["environment"] != self.environment:
            self.reuse_status = "different_model_deployment_or_runtime"
            return []
        prior_result = read(self.reuse / "result.json")
        self.reuse_status = ("considered" if prior.get("transport_samples", {}) == self.transport_samples.selection
                             else "transport_profile_changed")
        return [ExplorationAsset(SourceProductions(Path(row["production"])), Path(row["costs"]))
                for row in prior_result["assets"]]

    def prepare_sources(self, pending):
        directory = self.root / "plan"
        if not (directory / "plan.json").is_file():
            pairs = [(target.identity["configuration"], target.identity["workload"]) for target in pending]
            with tempfile.TemporaryDirectory(prefix=".planning-", dir=self.root) as temporary:
                staged = Path(temporary) / "plan"
                PlanningProcess(self.planner, staged, service_evidence=True).run(targets=pairs)
                staged.rename(directory)

        if not (self.training / "result.json").is_file():
            allowance = self.allowance(self.training)
            if allowance["postprocessing_budget_seconds"] <= 0:
                raise AllowanceExhausted("no allowance to compile missing templates")
            SourceAcquisition(SimpleNamespace(
                **allowance, output=self.training, plan=directory, deployment=self.root / "inputs/deployment.json",
                observations=None, keep_intermediates=True, resource_trace=self.args.details,
            ), self.place, self.limit).run(allow_incomplete=True)
        return SourceProductions(self.training)

    def prepare_costs(self, generation, fresh, prior_jobs, gaps, previous):
        directory = self.cost_task / str(generation)
        costs = directory / "costs.json"
        if costs.is_file():
            return costs
        allowance = self.allowance(directory)
        if allowance["postprocessing_budget_seconds"] <= 0:
            raise AllowanceExhausted("no allowance to prepare costs")
        arguments = SimpleNamespace(
            **allowance, output=directory, production=[fresh.directory] if fresh else [],
            samples=[], demand=[], gap=gaps, keep_intermediates=self.args.details,
        )
        CostAcquisition(arguments, self.place, self.reuse_samples, self.limit, transport_samples=self.transport_samples,
                        prior_jobs=prior_jobs, prior_costs=previous).run()
        return costs

    def predict(self, selected, runtime, first_costs, rows):
        costs, generation = first_costs, 0
        attempted = set()
        reasons = CostRequirements.reported_reasons(read(costs)["demand"])

        while selected:
            pending, jobs = {}, defaultdict(set)
            stage = f"prepared-{generation}" if generation else "prepared"
            for target, library in selected.items():
                row = target.predict(runtime, self.limit, ExplorationAsset(library, costs), stage)
                rows[target] = row
                if row["status"] != "cost_gap":
                    continue
                path = self.root / row["evidence"]
                point = WorkPoint.from_gap(read(path))
                if point in reasons or point in attempted:
                    reason = reasons.get(point, "covered_query_failed_in_execution")
                    row["sampling"] = {"costs": str(costs), "reason": reason}
                    continue
                pending[target] = point, path
                jobs[library.directory].update(library.jobs_for(target.input))
            if not pending:
                return

            # Only newly reached coordinates request more evidence. Updated predictions start from initial state.
            generation += 1
            try:
                costs = self.prepare_costs(generation, None, jobs, [path for _, path in pending.values()], [costs])
            except AllowanceExhausted as failure:
                for target in pending:
                    rows[target]["sampling"] = {"reason": "budget_exhausted", "message": str(failure)}
                return
            report = read(costs)["demand"]
            reasons = CostRequirements.reported_reasons(report)
            ready = {}
            for target, (point, path) in pending.items():
                attempted.add(point)
                if report["targets"][str(path)]["counts"].get("covered"):
                    ready[target] = selected[target]
                else:
                    rows[target]["sampling"] = {"costs": str(costs), "reason": reasons[point]}
            selected = ready

    def explore(self):
        previous = self.previous()
        runtime = {"binary": self.binary, "control_image": self.runtime["control_image"],
                   "modeling_image": self.runtime["modeling_image"], "timeout": self.args.timeout,
                   "execution_trace": self.args.details}
        targets = [ExplorationTarget(self.root, index, self.source["model"], configuration, workload, self.predictor)
                   for index, (configuration, workload) in enumerate(
                       product(self.planner.configurations.values(), self.planner.workloads.values()))]
        rows, selected, missing_templates = {}, {}, []
        existing_jobs, gaps = defaultdict(set), []

        for target in targets:
            match = target.reuse(previous, runtime, self.limit)
            if match is None:
                missing_templates.append(target)
                continue
            asset, row = match
            refresh_costs = self.reuse_status == "transport_profile_changed"
            rows[target] = target.identity if refresh_costs else row
            if row["status"] == "template_gap":
                missing_templates.append(target)
            elif refresh_costs or row["status"] == "cost_gap":
                selected[target] = asset.sources
                existing_jobs[asset.production].update(asset.sources.jobs_for(target.input))
                if row["status"] == "cost_gap":
                    gaps.append(self.root / row["evidence"])

        fresh = None
        if missing_templates:
            try:
                fresh = self.prepare_sources(missing_templates)
            except AllowanceExhausted as failure:
                for target in missing_templates:
                    rows[target] = {**rows.get(target, target.identity),
                                    "status": "budget_exhausted", "message": str(failure)}
            else:
                for target in missing_templates:
                    if fresh.select(target.input) is None:
                        rows[target] = {**rows.get(target, target.identity), **fresh.failure(target.input)}
                    else:
                        selected[target] = fresh

        if selected:
            try:
                training = fresh if fresh in selected.values() else None
                prior_costs = list(dict.fromkeys(asset.costs for asset in previous))
                costs = self.prepare_costs(0, training, existing_jobs, gaps, prior_costs)
            except AllowanceExhausted as failure:
                for target in selected:
                    rows[target] = {**rows.get(target, target.identity),
                                    "status": "budget_exhausted", "message": str(failure)}
            else:
                self.predict(selected, runtime, costs, rows)

        if fresh and not self.args.details:
            SourceAcquisition.release_intermediates(self.training)

        # Keep newly prepared assets first and publish only the bundles serving a final target outcome.
        assets = {}
        for row in sorted(rows.values(), key=lambda row: not row.get("asset_use", "").startswith("prepared")):
            if "asset" in row:
                asset = row["asset"]
                assets[asset["production"], asset["costs"]] = asset
        usage = {**Usage.total(self.spending().values()).document(), "retained_bytes": retained_bytes(self.root)}
        return ExplorationReport([rows[target] for target in targets], list(assets.values()), usage, self.reuse_status)

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
        return read(result)["status"] == "complete"
