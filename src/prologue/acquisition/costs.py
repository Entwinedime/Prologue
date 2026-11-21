"""Publish one bounded set of costs using observed graphs and native host recipes."""

import fcntl
import filecmp
import json
import math
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.ascend.host_samples import HostCostSamples
from prologue.acquisition.capture import CaptureProcess
from prologue.acquisition.production import CaptureProduction
from prologue.acquisition.recording import PublishedCapture
from prologue.integrations.runtimes.docker import ModelingCommand, OwnedContainers
from prologue.performance.requirements import CostRequirements
from prologue.performance.samples import WorkCatalog
from prologue.acquisition.fitting import FitCosts
from prologue.workflow.limits import AllowanceExhausted, OperationLimit, Usage
from prologue.acquisition.publication import SourceProductions


def read(path):
    return json.loads(path.read_text())


class CostSource:
    def __init__(self, production, capture, job, deployment, output):
        self.production, self.output = production, output
        self.deployment = deployment
        self.recording = PublishedCapture(capture, job)
        if read(production / "production.json")["inputs"]["capture_path"] != str(self.recording.source):
            raise ValueError("training production belongs to a different native observation")
        self.ranks = read(production / "result.json")["observed_roles"]
        self.projection = production
        self.recipes = {recipe.name: recipe for recipe in HostCostSamples.recipes(job)}
        self.host_points = set()
        self.cold = False


class CostAcquisition:
    def __init__(self, args, place, reuse_samples, limit=None, *, transport_samples, prior_jobs=None, prior_costs=()):
        self.args = args
        self.place = place
        self.reuse_samples = reuse_samples
        self.output = args.output.resolve()
        self.binary = args.binary.resolve(strict=True)
        if any(not math.isfinite(value) or value < 1e-9 for value in (args.timeout, args.postprocessing_budget_seconds)):
            raise ValueError("cost acquisition needs positive finite time allowances")
        if (not math.isfinite(args.native_budget_seconds) or args.native_budget_seconds < 0
                or args.attempt_limit < 0 or args.storage_bytes <= 0):
            raise ValueError("cost acquisition needs nonnegative native allowances and positive storage")
        self.budget = Usage(int(args.native_budget_seconds * 1e9), args.attempt_limit,
                            int(args.postprocessing_budget_seconds * 1e9))
        self.limit = limit if limit is not None else OperationLimit(self.output, args.storage_bytes)
        self.images = {
            role: subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
            for role, image in (("source", args.source_image), ("modeling", args.modeling_image))
        }

        collections = list(dict.fromkeys(path.resolve(strict=True) for path in args.production))
        self.collections = collections
        self.prior_jobs = prior_jobs or {}
        self.sources = {}
        self.demand = CostRequirements()
        paths = list(args.samples)
        for path in prior_costs:
            asset = read(path)
            paths.extend(Path(name) for name in asset["sources"] if name not in asset.get("transport_samples", {}))

            # Preserve cross-role bindings by requalifying their coordinates against the original measured rows.
            for sample in asset["reused"]:
                self.demand.targets[str(path)] = "previously_qualified_role_work"
                self.demand.samples(str(path), sample)
        self.samples = {path: read(path) for path in dict.fromkeys(path.resolve(strict=True) for path in paths)}
        if any("transport_profile" in source for source in self.samples.values()):
            raise ValueError("select physical peer service profiles with --transport-samples")
        self.samples.update(transport_samples.sources)
        self.transport_samples = transport_samples.selection
        self.gaps = {path.resolve(strict=True): read(path) for path in args.gap}
        self.graphs = {graph.resolve(strict=True): identities.resolve(strict=True) for graph, identities in args.demand}
        self.inputs = {
            "collections": [str(path) for path in collections], "images": self.images,
            "samples": [str(path) for path in self.samples], "gaps": [str(path) for path in self.gaps],
            "demand": [[str(graph), str(identities)] for graph, identities in self.graphs.items()],
        }
        if self.prior_jobs:
            self.inputs["prior_jobs"] = {str(path): sorted(jobs) for path, jobs in self.prior_jobs.items()}
        if prior_costs:
            self.inputs["prior_costs"] = [str(path) for path in prior_costs]
        if self.transport_samples:
            self.inputs["transport_samples"] = self.transport_samples

    def load_sources(self, collection, selected=None):
        bundle = SourceProductions(collection)
        for identity, path in bundle.paths.items():
            if selected is not None and identity not in selected:
                continue
            capture = bundle.captures[identity].resolve(strict=True)
            production = path.resolve(strict=True)
            if any(self.output.is_relative_to(root) for root in (collection, production, capture)):
                raise ValueError("cost acquisition must own an output outside its training sources")
            if production not in self.sources:
                directory = self.output / "sources" / str(len(self.sources))
                source = CostSource(production, capture, bundle.jobs[str(identity)], bundle.deployment, directory)
                self.sources[production] = source
                yield source

    def save(self):
        write_json(self.output / "execution.json", self.state, indent=2)

    def prepare(self):
        executable = self.output / "prologue"
        state_path = self.output / "execution.json"
        if state_path.exists():
            self.state = read(state_path)
            if self.state["inputs"] != self.inputs or not filecmp.cmp(self.binary, executable, shallow=False):
                raise ValueError("cost acquisition inputs or executable changed")
            if self.state["container"] is not None:
                OwnedContainers([self.state["container"]]).close()
                self.state["container"] = None
                self.save()
            return
        if any(path.name != ".lock" for path in self.output.iterdir()):
            raise ValueError("new cost acquisition requires an empty directory")
        shutil.copyfile(self.binary, executable)
        executable.chmod(0o500)
        self.state = {"inputs": self.inputs, "postprocessing_ns": 0, "container": None}
        self.save()

    @contextmanager
    def processing(self, container=None):
        spent = self.state["postprocessing_ns"]
        remaining = self.budget.processing_ns - spent - self.projection_usage(self.output).processing_ns
        if remaining <= 0:
            raise AllowanceExhausted("cost postprocessing allowance exhausted")

        # Reserve before launch so interruption cannot make unfinished work free.
        self.state.update(postprocessing_ns=spent + remaining, container=container)
        self.save()
        started = time.monotonic_ns()
        try:
            self.limit.check()
            yield remaining / 1e9
            self.limit.check()
            if time.monotonic_ns() - started > remaining:
                raise AllowanceExhausted("cost postprocessing allowance exhausted")
        finally:
            if container is not None:
                OwnedContainers([container]).close()
            self.state.update(postprocessing_ns=spent + time.monotonic_ns() - started, container=None)
            self.save()

    def command(self, output, command):
        if output.is_file():
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        name = command.name if isinstance(command, ModelingCommand) else None
        with self.processing(name) as allowance, output.with_suffix(".log").open("a") as log:
            arguments = command.arguments(allowance) if name else command
            result = self.limit.run(arguments, log, allowance)
            if result:
                raise subprocess.CalledProcessError(result, arguments)

    def run_model(self, operation, output, *inputs):
        command = ModelingCommand(self.output / "prologue", self.images["modeling"], operation, output, *inputs)
        self.command(output, command)

    def samples_from(self, output, action, **inputs):
        command = [sys.executable, "-m", "prologue.app", action]
        for name, value in {**inputs, "output": output}.items():
            command.extend(("--" + name.replace("_", "-"), str(value)))
        self.command(output, command)
        sample = read(output)
        self.samples[output] = sample
        return sample

    def project(self, source):
        if all((source.production / "work" / f"{rank}-{file}").is_file()
               for rank in source.ranks for file in ("graph.json", "facts.json", "sdk/trace.json")):
            return

        # A compact parent still owns the closed observation. Recreate only derived evidence in this task.
        source.projection = source.output / "projection"
        allowance = self.budget.available_to(CaptureProduction.usage(source.projection), self.spending(self.output))
        if allowance.processing_ns <= 0:
            raise AllowanceExhausted("cost projection allowance exhausted")
        CaptureProduction(SimpleNamespace(
            output=source.projection, binary=self.output / "prologue", product="graphs",
            budget_seconds=allowance.processing_ns / 1e9, source_image=self.images["source"],
            modeling_image=self.images["modeling"], keep_intermediates=True,
        ), source.recording, self.limit).run()

    def inspect(self, source, requirements):
        self.project(source)
        external = HostCostSamples.external_work(source.recording.source)
        for rank in source.ranks:
            work = source.projection / "work"
            graph = work / f"{rank}-graph.json"
            identities = source.output / f"{rank}-work.json"
            self.run_model("work-identities", identities, graph)
            with self.processing():
                catalog = WorkCatalog(graph, identities)
                for point in requirements.graph(str(graph), catalog):
                    definition = requirements.definitions[point.definition]
                    interval = definition["work"][point.interval]
                    if HostCostSamples.supports(interval, definition["resources"][interval["resource"]]):
                        source.host_points.add(point)
                for identity, kind, features in external[rank]:
                    source.host_points.add(requirements.external(str(graph), catalog.domain, identity, kind, features))

            # The same detailed observation must not contribute device durations twice.
            observation = {"production": str(source.production), "rank": rank}
            if not any(sample.get("observation") == observation for sample in self.samples.values()):
                projection = {"projection": source.projection} if source.projection != source.production else {}
                self.samples_from(source.output / f"{rank}-device.json", "cost-samples", production=source.production,
                                  rank=rank, work_identities=identities, **projection)

    @staticmethod
    def projection_usage(directory):
        return Usage.total(CaptureProduction.usage(path.parent) for path in directory.glob("sources/*/projection/production.json"))

    @classmethod
    def spending(cls, directory):
        state = directory / "execution.json"
        processing = Usage(processing_ns=read(state)["postprocessing_ns"]) if state.is_file() else Usage()
        native = (CaptureProcess.usage(path.parent) for path in directory.glob("sources/*/*/execution.json"))
        return Usage.total((processing, cls.projection_usage(directory), *native))

    def measure(self, source, recipe, allowance_ns):
        output = source.output / recipe
        origin = source.recording.inputs
        launch = self.place(source.recipes[recipe].job, source.deployment)
        launch.require_recorded_placement(origin["deployment"])
        CaptureProcess(SimpleNamespace(
            output=output, timeout=self.args.timeout,
            budget_seconds=allowance_ns / 1e9, attempt_limit=1, source_image=origin["image"], resource_trace=False,
        ), launch, self.limit, host_timing=True).run()
        for rank in source.ranks:
            work = source.projection / "work"
            sample = self.samples_from(
                source.output / f"{recipe}-{rank}-host.json", "host-cost-samples", detailed=source.recording.directory, host=output,
                facts=work / f"{rank}-facts.json", trace=work / f"{rank}-sdk/trace.json",
                graph=work / f"{rank}-graph.json", work_identities=source.output / f"{rank}-work.json",
            )
            source.cold |= bool(sample["qualification"]["excluded_cold_scopes"])

    def fit(self):
        return FitCosts.qualify(self.samples, self.demand, self, self.output, self.reuse_samples, self.processing)

    def acquire(self):
        for collection in self.collections:
            for source in self.load_sources(collection):
                self.inspect(source, self.demand)
        with self.processing():
            for graph, identities in self.graphs.items():
                if str(graph) not in self.demand.targets:
                    self.demand.graph(str(graph), WorkCatalog(graph, identities))
            for path, gap in self.gaps.items():
                self.demand.gap(str(path), gap)

        fitted, reused, counts, missing = self.fit()
        if missing and self.prior_jobs:

            # Prior graphs supply possible measurements. Their unrelated work is outside the requested demand.
            for collection, jobs in self.prior_jobs.items():
                for source in self.load_sources(collection, jobs):
                    self.inspect(source, CostRequirements())
            fitted, reused, counts, missing = self.fit()

        candidates = {(source, recipe): CaptureProcess.attempts(source.output / recipe)
                      for source in self.sources.values() for recipe in source.recipes}
        explicit = {str(path) for path in (*self.graphs, *self.gaps)}
        focus = {point for point, targets in self.demand.points.items() if explicit.intersection(targets)}
        evidence = []
        deferred = set()
        while candidates and missing:

            # Prefer caller-supplied demand when training-only points compete for a native attempt.
            requested = {source: source.host_points & missing.keys() for source in self.sources.values()}
            applicable = [(source, recipe) for source, recipe in candidates if requested[source]
                          and (source.cold or not source.recipes[recipe].requires_cold)]
            if not applicable:
                break
            source, recipe = max(applicable, key=lambda key: (
                bool(candidates[key]), len(requested[key[0]] & focus), len(requested[key[0]]),
            ))
            output = source.output / recipe
            previous = candidates[source, recipe]
            spent = self.spending(self.output)
            allowance = max(0, self.budget.native_ns - spent.native_ns)
            if not previous and (not allowance or spent.attempts >= self.budget.attempts):
                deferred.update(*(candidate.host_points for candidate, _ in applicable))
                break

            del candidates[source, recipe]
            row = {"capture": str(output.relative_to(self.output)), "candidate_points": len(requested[source])}
            evidence.append(row)
            if previous and previous[-1]["status"] == "failed":
                row["status"] = "capture_failed"
                continue
            try:
                self.measure(source, recipe, allowance + sum(attempt["charged_ns"] for attempt in previous))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
                attempts = CaptureProcess.attempts(output)
                if not attempts or attempts[-1]["status"] != "failed":
                    raise
                self.limit.check()
                row["status"] = "capture_failed"
                continue
            before = missing
            fitted, reused, counts, missing = self.fit()
            row.update(status="measured", additional_points=len(before.keys() - missing.keys()))

        observed = set().union(*(source.host_points for source in self.sources.values()))
        reasons = {
            point: ("native_budget_exhausted" if point in deferred else
                    "no_qualified_measurement" if point in observed else "no_observed_recipe")
            for point in missing
        }
        return {
            "domains": fitted.domains, "sources": [str(path) for path in self.samples], "reused": reused,
            "transport_samples": self.transport_samples,
            "demand": self.demand.report(counts, missing, reasons), "recipes": evidence,
            **self.spending(self.output).document(), "retained_bytes": self.limit.bytes_used(),
            "status": "covered" if not missing else "native_budget_exhausted" if deferred else "uncovered",
        }

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.prepare()
            result = self.output / "costs.json"
            try:
                if not result.is_file():
                    write_json(result, self.acquire(), replace=False)
            finally:
                write_json(self.output / "status.json", self.spending(self.output).document(), indent=2)
        print(f"Created {result}")
