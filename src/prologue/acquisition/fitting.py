"""Qualify training samples against required work before publishing fitted costs."""

import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from statistics import median

from prologue.artifacts import write_json
from prologue.integrations.runtimes.docker import ModelingProcess
from prologue.performance.fitting import FittedCosts
from prologue.performance.requirements import CostRequirements
from prologue.performance.samples import WorkCatalog


class FitCosts:
    def __init__(self, arguments, reuse_samples):
        self.arguments = arguments
        self.reuse_samples = reuse_samples

    @staticmethod
    def qualify(sources, demand, model, directory, reuse_samples, accounting=nullcontext):
        with accounting():
            fitted = FittedCosts(sources.values())
        counts, missing = demand.check(model, fitted.domains, directory, accounting)
        with accounting():
            reused = reuse_samples(demand, missing, sources)
            if reused:
                fitted = FittedCosts([*sources.values(), *reused])
        if reused:
            counts, missing = demand.check(model, fitted.domains, directory, accounting)
        return fitted, reused, counts, missing

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    @staticmethod
    def assess(source, estimates):
        counts, errors = defaultdict(Counter), defaultdict(list)
        for row, (status, estimate) in zip(source["samples"], estimates, strict=True):
            kind = source["definitions"][row["definition"]]["work"][row["interval"]]["kind"]
            observations = row["observations_ns"]
            counts[kind][status] += len(observations)
            if estimate is not None and (actual := median(observations)) > 0:
                errors[kind].append(abs(estimate - actual) / actual)
        return {
            "counts": counts,
            "median_point_relative_error": {kind: median(values) for kind, values in sorted(errors.items())},
            "unmeasured": source["unmeasured"],
        }

    def run(self):
        sources = {path.resolve(strict=True): self.read(path) for path in self.arguments.samples}
        demand = CostRequirements()
        for graph_path, identities_path in self.arguments.demand:
            demand.graph(str(graph_path.resolve()), WorkCatalog(graph_path, identities_path))
        for path in self.arguments.gap:
            demand.gap(str(path.resolve()), self.read(path))

        checks = CostRequirements()
        check_sources = []
        for path in self.arguments.check:
            source = self.read(path)
            name = str(path.resolve())
            check_sources.append((name, source, checks.samples(name, source)))

        directory = self.arguments.output.parent.resolve()
        model = None
        if demand.points or checks.points:
            if self.arguments.binary is None:
                raise ValueError("cost demand and sample checks require --binary")
            directory.mkdir(parents=True, exist_ok=True)
            model = ModelingProcess(self.arguments.binary, self.arguments.modeling_image, self.arguments.timeout)

        fitted, reused, counts, missing = self.qualify(sources, demand, model, directory, self.reuse_samples)
        estimates = checks.query(model, fitted.domains, directory)
        evaluations = [{"source": name, **self.assess(source, (estimates[point] for point in points))}
                       for name, source, points in check_sources]

        write_json(self.arguments.output, {
            "domains": fitted.domains,
            "sources": [str(path) for path in sources],
            "source_gaps": [source["unmeasured"] for source in sources.values()],
            "reused": reused,
            "checks": evaluations,
            "demand": demand.report(counts, missing),
        }, replace=False)
