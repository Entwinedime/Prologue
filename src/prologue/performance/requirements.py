"""Collect the same work coordinates from projected graphs and prediction gaps."""

import json
from collections import ChainMap, Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.artifacts import write_json
from prologue.performance.models import ModelBindings


def cost_identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_work_interval(query):
    work, interval = query["definition"]["work"], query["interval"]
    if type(interval) is not int or not 0 <= interval < len(work) or work[interval]["kind"] != query["work"]:
        raise ValueError("work query must identify a declared interval of the requested kind")


@dataclass(frozen=True, slots=True)
class WorkPoint:
    domain: str
    definition: str
    interval: int
    phase: str
    axes: tuple
    transport: str | None = None

    @classmethod
    def at(cls, domain, definition, interval, features):
        transport = cost_identity(features["transport"]) if "transport" in features else None
        return cls(domain, definition, interval, features["phase"], tuple(sorted(features["axes"].items())), transport)

    def features(self):
        result = {"phase": self.phase, "axes": dict(self.axes)}
        if self.transport is not None:
            result["transport"] = json.loads(self.transport)
        return result


class CostRequirements:
    """Share identical observed points while retaining the targets that need them."""

    def __init__(self):
        self.domains = {}
        self.definitions = {}
        self.targets = {}
        self.points = defaultdict(Counter)

    @staticmethod
    def intern(table, value):
        identity = cost_identity(value)
        table.setdefault(identity, value)
        return identity

    def add(self, target, domain, definition, interval, features):
        point = WorkPoint.at(domain, definition, interval, features)
        self.points[point][target] += 1
        return point

    def graph(self, target, catalog):
        for _ in self.observe_graph(target, catalog):
            pass

    def observe_graph(self, target, catalog):
        self.targets[target] = "observed_graph"
        domain = self.intern(self.domains, catalog.domain)
        identities = [self.intern(self.definitions, value) for value in catalog.definitions]
        for work, definition, interval, features in catalog:
            yield work, self.add(target, domain, identities[definition], interval, features)

    def external(self, target, domain, identity, kind, features):
        domain = self.intern(self.domains, domain)
        definition = self.intern(self.definitions, {"external": identity, "work": [{"kind": kind}]})
        return self.add(target, domain, definition, 0, features)

    def gap(self, target, gap):
        require_work_interval(gap)
        self.targets[target] = "first_prediction_gap"
        domain = self.intern(self.domains, gap["domain"])
        definition = self.intern(self.definitions, gap["definition"])
        self.add(target, domain, definition, gap["interval"], gap["features"])

    def samples(self, target, source, *, basis):
        self.targets[target] = basis
        domain = self.intern(self.domains, source["domain"])
        definitions = [self.intern(self.definitions, value) for value in source["definitions"]]
        return [self.add(target, domain, definitions[row["definition"]], row["interval"], row["features"])
                for row in source["samples"]]

    def query(self, model, domains, directory, accounting, held_out):
        requests = (self, held_out) if held_out is not None else (self,)
        if not any(request.points for request in requests):
            return {}
        with TemporaryDirectory(prefix=".cost-points-", dir=directory) as scratch:
            scratch = Path(scratch)
            points_path, asset_path, result_path = (scratch / name for name in ("points.json", "costs.json", "estimates.json"))
            with accounting():
                groups = defaultdict(dict)
                for request in requests:
                    for point in request.points:
                        groups[point.domain][point] = None
                source_domains = ChainMap(*(request.domains for request in requests))
                declared = ChainMap(*(request.definitions for request in requests))

                queries = []
                for domain, points in groups.items():
                    definitions = {}
                    rows = [[definitions.setdefault(point.definition, len(definitions)), point.interval, point.features()]
                            for point in points]
                    queries.append({
                        "domain": source_domains[domain], "definitions": [declared[key] for key in definitions],
                        "points": rows,
                    })
                write_json(points_path, queries)
                write_json(asset_path, {"domains": domains})

            model.run_model("cost-points", result_path, points_path, asset_path)
            with accounting():
                estimates = json.loads(result_path.read_text())
                return {point: estimate for points, results in zip(groups.values(), estimates, strict=True)
                        for point, estimate in zip(points, results, strict=True)}

    def check(self, model, domains, directory, accounting=nullcontext, *, held_out=None):
        estimates = self.query(model, domains, directory, accounting, held_out)
        with accounting():

            # Held-out points share the evaluator. Only training demand can trigger reuse or a native measurement.
            counts = {target: Counter() for target in self.targets}
            missing = {}
            for point, targets in self.points.items():
                status, _ = estimates[point]
                for target, number in targets.items():
                    counts[target][status] += number
                if status != "covered":
                    missing[point] = status
            return counts, missing, estimates if held_out is not None and held_out.points else {}

    def report(self, counts, missing, reasons=None):
        domains = {}
        for point, status in missing.items():
            domain = domains.setdefault(point.domain, {"definitions": {}, "work": {}})
            definitions = domain["definitions"]
            index = definitions.setdefault(point.definition, len(definitions))
            key = index, point.interval, point.phase, status
            definition = self.definitions[point.definition]
            row = domain["work"].setdefault(key, {
                "definition": index, "interval": point.interval, "phase": point.phase, "status": status,
                "kind": definition["work"][point.interval]["kind"], "points": [],
                "candidate_models": [family.family for family in ModelBindings.for_interval(definition, point.interval).candidates],
            })
            value = point.features()
            del value["phase"]
            value["targets"] = dict(self.points[point])
            if reasons is not None:
                value["reason"] = reasons[point]
            row["points"].append(value)
        return {
            "targets": {target: {"basis": basis, "counts": dict(counts[target])} for target, basis in self.targets.items()},
            "uncovered": [
                {"domain": self.domains[domain], "definitions": [self.definitions[name] for name in group["definitions"]],
                 "work": list(group["work"].values())}
                for domain, group in domains.items()
            ],
            "scope": "Recorded graph and protocol work and each supplied first cost gap. Work on unreached control paths is unknown.",
        }

    @staticmethod
    def reported_reasons(report):
        reasons = {}
        for group in report["uncovered"]:
            domain = cost_identity(group["domain"])
            definitions = [cost_identity(definition) for definition in group["definitions"]]
            for work in group["work"]:
                identity = definitions[work["definition"]]
                for row in work["points"]:
                    point = WorkPoint.at(domain, identity, work["interval"], {**row, "phase": work["phase"]})
                    reasons[point] = row["reason"]
        return reasons
