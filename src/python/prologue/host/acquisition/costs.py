"""Acquire missing work coordinates from existing evidence or a bounded native measurement."""

import math
import subprocess
import time
from contextlib import contextmanager

from prologue.common.cost import WorkCatalog
from prologue.common.io import discard_path, read_json, write_json
from prologue.host.acquisition.adapters import CostSampleAdapter
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.fitting import FitCosts
from prologue.host.acquisition.options import (
    CostAcquisitionOptions,
    DeviceSampleOptions,
    LocalAcquisitionOptions,
    ProductionOptions,
)
from prologue.host.acquisition.production import CaptureProduction
from prologue.host.acquisition.recording import PublishedCapture, SourceProductions
from prologue.host.acquisition.rules import EvidenceGap
from prologue.host.performance.requirements import CostRequirements
from prologue.host.runtime.docker import DockerImage, ModelingCommand, OwnedContainers
from prologue.host.runtime.limits import AllowanceExhausted, OperationLimit, Usage, retained_bytes
from prologue.host.runtime.workspace import BinarySnapshot, TaskWorkspace, file_lock


class CostSource:
    def __init__(self, production, capture, job, deployment, output):
        self.production, self.output = production, output
        self.deployment = deployment
        self.recording = PublishedCapture(capture, job)
        if read_json(production / "production.json")["inputs"]["capture_path"] != str(self.recording.source):
            raise ValueError("training production belongs to a different native observation")
        self.ranks = read_json(production / "result.json")["observed_roles"]
        self.projection = production

    def identities(self, rank):
        return self.output / "work" / f"{rank}-work.json"

    @staticmethod
    def release_intermediates(output):
        discard_path(output / "work")
        CaptureProduction.retire_projection(output / "projection")


class CostAcquisition:
    def __init__(
        self,
        args: CostAcquisitionOptions,
        place,
        reuse_samples,
        *,
        transport_samples,
        sample_adapter: type[CostSampleAdapter],
    ):
        self.sample_adapter = sample_adapter
        self.args = args
        self.place = place
        self.reuse_samples = reuse_samples
        self.output = args.output.resolve()
        self.binary = args.binary.resolve(strict=True)
        durations = args.timeout, args.postprocessing_budget_seconds
        if any(not math.isfinite(value) or value < 1e-9 for value in durations):
            raise ValueError("cost acquisition needs positive finite time allowances")
        if (not math.isfinite(args.native_budget_seconds) or args.native_budget_seconds < 0
                or args.attempt_limit < 0 or args.storage_bytes <= 0):
            raise ValueError("cost acquisition needs nonnegative native allowances and positive storage")
        self.budget = Usage(int(args.native_budget_seconds * 1e9), args.attempt_limit,
                            int(args.postprocessing_budget_seconds * 1e9))
        self.limit = OperationLimit(self.output, args.storage_bytes)
        self.images = {
            role: DockerImage.id(image)
            for role, image in (("source", args.source_image), ("modeling", args.modeling_image))
        }

        collections = list(dict.fromkeys(path.resolve(strict=True) for path in args.production))
        self.collections = collections
        self.sources = {}
        self.demand = CostRequirements()
        self.samples = dict(args.samples)
        if any("transport_profile" in source for source in self.samples.values()):
            raise ValueError("select physical peer service profiles with --transport-samples")
        self.samples.update(transport_samples.sources)
        self.transport_samples = transport_samples.selection
        self.gaps = {path.resolve(strict=True): read_json(path) for path in args.gap}
        self.graphs = {graph.resolve(strict=True): identities.resolve(strict=True) for graph, identities in args.demand}
        self.inputs = {
            "collections": [str(path) for path in collections], "images": self.images,
            "samples": [str(path) for path in self.samples], "gaps": [str(path) for path in self.gaps],
            "demand": [[str(graph), str(identities)] for graph, identities in self.graphs.items()],
            "keep_intermediates": args.keep_intermediates,
        }
        if self.transport_samples:
            self.inputs["transport_samples"] = self.transport_samples

    def load_sources(self, collection):
        bundle = SourceProductions(collection)
        for identity, path in bundle.paths.items():
            capture = bundle.captures[identity].resolve(strict=True)
            production = path.resolve(strict=True)
            if any(self.output.is_relative_to(root) for root in (collection, production, capture)):
                raise ValueError("cost acquisition must own an output outside its training sources")
            if production not in self.sources:
                directory = self.output / "sources" / str(len(self.sources))
                job = bundle.jobs[str(identity)]
                source = CostSource(production, capture, job, bundle.deployment, directory)
                source.sampling = self.place(job, bundle.deployment).sampling(source)
                self.sources[production] = source
                yield source

    def save(self):
        write_json(self.output / "execution.json", self.state, indent=2)

    def prepare(self):
        snapshot = BinarySnapshot(self.binary)
        with TaskWorkspace(self.output).initialize("execution.json") as draft:
            if draft is not None:
                snapshot.copy_to(draft / "prologue")
                self.state = {"inputs": self.inputs, "postprocessing_ns": 0, "container": None}
                write_json(draft / "execution.json", self.state, indent=2)
                return
        self.state = read_json(self.output / "execution.json")
        if self.state["inputs"] != self.inputs:
            raise ValueError("cost acquisition inputs changed")
        snapshot.check(self.output / "prologue")
        if self.state["container"] is not None:
            OwnedContainers([self.state["container"]]).close()
            self.state["container"] = None
            self.save()

    @contextmanager
    def processing(self, container=None):
        spent = self.state["postprocessing_ns"]
        remaining = self.budget.processing_ns - spent - self.projection_usage(self.output).processing_ns
        if remaining <= 0:
            raise AllowanceExhausted("cost postprocessing allowance exhausted")

        self.state["container"] = container
        with TaskWorkspace.reserve(self.state, "postprocessing_ns", remaining, self.save,
                                   settled=lambda: self.state["container"] is None) as deadline:
            try:
                self.limit.check()
                yield remaining / 1e9
                self.limit.check()
                if time.monotonic_ns() > deadline:
                    raise AllowanceExhausted("cost postprocessing allowance exhausted")
            finally:
                if container is not None:
                    OwnedContainers([container]).close()
                self.state["container"] = None

    def command(self, output, command):
        if output.is_file():
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        with self.processing(command.name) as allowance, output.with_suffix(".log").open("a") as log:
            arguments = command.arguments(allowance)
            result = self.limit.run(arguments, log, allowance)
            if result:
                raise subprocess.CalledProcessError(result, arguments)

    def run_model(self, operation, output, *inputs):
        command = ModelingCommand(self.output / "prologue", self.images["modeling"], operation, output, *inputs)
        self.command(output, command)

    def sample(self, output, operation, options):
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            with self.processing() as seconds:
                self.limit.run_worker(operation, options, output.with_suffix(".log"), seconds)

    def project(self, source):
        if all((source.production / "work" / f"{rank}-{file}").is_file()
               for rank in source.ranks for file in ("graph.json", "facts.json", "sdk/trace.json")):
            return

        # A compact parent still owns the closed observation. Recreate only derived evidence in this task.
        source.projection = source.output / "projection"
        allowance = self.budget.available_to(CaptureProduction.usage(source.projection), self.spending(self.output))
        if allowance.processing_ns <= 0:
            raise AllowanceExhausted("cost projection allowance exhausted")
        CaptureProduction(ProductionOptions(
            output=source.projection, binary=self.output / "prologue", product="graphs",
            budget_seconds=allowance.processing_ns / 1e9, source_image=self.images["source"],
            modeling_image=self.images["modeling"], keep_intermediates=True,
        ), source.recording, self.place, self.limit).run()

    def inspect(self, source, requirements):
        self.project(source)
        work = source.projection / "work"
        needed = {}
        for rank in source.ranks:
            self.run_model("work-identities", source.identities(rank), work / f"{rank}-graph.json")
            observation = {"production": str(source.production), "rank": rank}
            if not any(sample.get("observation") == observation for sample in self.samples.values()):
                needed[rank] = source.output / f"{rank}-device.json"

        pending = {rank: path for rank, path in needed.items() if not path.is_file()}
        if pending:
            self.export_device_samples(source, pending)

        for rank in source.ranks:
            with self.processing():
                if rank in needed:
                    path = needed[rank]
                    self.samples[path] = read_json(path)

                catalog = WorkCatalog(work / f"{rank}-graph.json", source.identities(rank))
                owner = f"{source.recording.source / 'capture.json'}#rank={rank}"
                source.sampling.observe(rank, owner, catalog, requirements)

    def export_device_samples(self, source, pending):
        options = DeviceSampleOptions(production=source.production, projection=source.projection, rank=tuple(pending),
                                      work_identities=tuple(source.identities(rank) for rank in pending),
                                      output=tuple(pending.values()))

        # Joint device samples share the clock calibration of one bounded worker.
        self.sample(options.output[-1], self.sample_adapter.device_samples, options)

    @staticmethod
    def projection_usage(directory):
        states = directory.glob("sources/*/projection/production.json")
        return Usage.total(CaptureProduction.usage(path.parent) for path in states)

    @classmethod
    def spending(cls, directory):
        state = directory / "execution.json"
        processing = Usage(processing_ns=read_json(state)["postprocessing_ns"]) if state.is_file() else Usage()
        journals = (*directory.glob("sources/*/*/execution.json"), *directory.glob("sources/*/*/native/execution.json"))
        native = (NativeAttempt.usage(path.parent) for path in journals)
        return Usage.total((processing, cls.projection_usage(directory), *native))

    def measure(self, recipe, output, requested, allowance_ns):
        arguments = LocalAcquisitionOptions(
            output=output, timeout=self.args.timeout, budget_seconds=allowance_ns / 1e9,
            attempt_limit=1, source_image=self.images["source"], binary=self.output / "prologue",
            modeling_image=self.images["modeling"],
        )
        recipe.collect(arguments, requested, self.limit, self.processing)
        self.samples.update(recipe.samples(output, self))

    def fit(self):
        return FitCosts.qualify(self.samples, self.demand, self, self.output, self.reuse_samples, self.processing)

    def acquire(self):
        explicit = bool(self.graphs or self.gaps)
        for collection in self.collections:
            for source in self.load_sources(collection):

                # A training graph locates native cuts. It supplies demand only for a cold, unscoped acquisition.
                self.inspect(source, CostRequirements() if explicit else self.demand)
        with self.processing():
            for graph, identities in self.graphs.items():
                if str(graph) not in self.demand.targets:
                    self.demand.graph(str(graph), WorkCatalog(graph, identities))
            for path, gap in self.gaps.items():
                self.demand.gap(str(path), gap)

        fitted, counts, missing, _ = self.fit()

        candidates = {recipe: source.output / recipe.name for source in self.sources.values()
                      for recipe in source.sampling.windows(self.demand, self.samples.values())}
        observed = set().union(*(recipe.points for recipe in candidates))
        evidence = []
        deferred = set()
        while candidates and missing:
            applicable = {recipe: missing.keys() & recipe.points for recipe in candidates if recipe.ready}
            applicable = {recipe: points for recipe, points in applicable.items() if points}
            if not applicable:
                break
            attempts = {
                recipe: NativeAttempt.attempts(candidates[recipe] / recipe.native_directory) for recipe in applicable
            }
            recipe = max(applicable, key=lambda candidate: (bool(attempts[candidate]), len(applicable[candidate])))
            output, requested, previous = candidates.pop(recipe), applicable[recipe], attempts[recipe]
            spent = self.spending(self.output)
            allowance = max(0, self.budget.native_ns - spent.native_ns)
            if not previous and (not allowance or spent.attempts >= self.budget.attempts):
                deferred.update(*applicable.values())
                break

            row = {"capture": str(output.relative_to(self.output)), "candidate_points": len(requested)}
            evidence.append(row)
            if previous and previous[-1]["status"] == "failed":
                row["status"] = "capture_failed"
                continue
            try:
                self.measure(recipe, output, requested, allowance + sum(attempt["charged_ns"] for attempt in previous))
            except EvidenceGap as gap:
                row.update(gap.report)
                continue
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
                failed = NativeAttempt.attempts(output / recipe.native_directory)
                if not failed or failed[-1]["status"] != "failed":
                    raise
                self.limit.check()
                row["status"] = "capture_failed"
                continue
            before = missing
            fitted, counts, missing, _ = self.fit()
            gained = len(before.keys() - missing.keys())
            row.update(status="measured" if gained else "no_new_coverage", additional_points=gained)

        reasons = {
            point: ("native_budget_exhausted" if point in deferred else
                    "no_qualified_measurement" if point in observed else "no_observed_recipe")
            for point in missing
        }
        return {
            **fitted.document(),
            "transport_samples": self.transport_samples,
            "demand": self.demand.report(counts, missing, reasons), "recipes": evidence,
            **self.spending(self.output).document(),
            "status": "covered" if not missing else "native_budget_exhausted" if deferred else "uncovered",
        }

    def run(self):
        with file_lock(self.output / ".lock"):
            self.prepare()
            result = self.output / "costs.json"
            try:
                if not result.is_file():
                    staged = self.output / ".fitted.json"
                    if not staged.is_file():
                        write_json(staged, self.acquire(), replace=False)

                    # A restart during cleanup needs only the durable fit and the immutable original samples.
                    if not self.args.keep_intermediates:
                        for directory in (self.output / "sources").glob("*"):
                            CostSource.release_intermediates(directory)
                    self.limit.check()
                    staged.rename(result)
            finally:
                write_json(self.output / "status.json", {
                    **self.spending(self.output).document(), "retained_bytes": retained_bytes(self.output),
                }, indent=2)
        print(f"Created {result}")
