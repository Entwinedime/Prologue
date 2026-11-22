"""Verify service costs without changing the observed serving execution."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.artifacts import write_json
from prologue.integrations.runtimes.docker import ModelingProcess

from .fixed_reference import FixedReference
from .latency_metrics import compare_request_metrics
from .reference import ReferenceRecording
from .source_evidence import ProductionEvidence


class FixedPerformance:
    def __init__(self, arguments):
        self.arguments = arguments
        self.output = arguments.output.resolve()
        self.report = {"capture": str(arguments.capture.resolve()), "reference": str(arguments.reference.resolve()),
                       "cost_asset": str(arguments.costs.resolve())}

    def finish(self, status):
        write_json(self.output / "result.json", {"status": status, **self.report}, indent=2, replace=False)
        return status == "compared"

    def estimate(self, reference, model, scratch):
        estimates, first_gaps = {}, {}
        for rank, (contents, _, _) in reference.case.sources.items():
            output = scratch / f"{rank}-costs.json"
            model.run_model("work-costs", output, reference.case.graphs / f"{rank}-graph.json", self.arguments.costs)
            result = json.loads(output.read_text())
            estimates[rank] = result["work_ns"]
            if len(estimates[rank]) != len(contents.graph["work"]):
                raise ValueError("fixed cost response changes the observed work set")
            if result["first_cost_gap"] is not None:
                first_gaps[rank] = result["first_cost_gap"]
        self.report["first_cost_gaps"] = first_gaps
        return [estimates[interval.rank][interval.index] for interval in reference.intervals]

    def assess(self, reference, estimates):
        coverage = defaultdict(lambda: {"work": 0, "measured": 0, "modeled": 0, "paired": 0,
                                        "signed_error_ns": 0, "absolute_error_ns": 0})
        first_missing = None
        for interval, estimate in zip(reference.intervals, estimates, strict=True):
            row = coverage[interval.piece["kind"]]
            duration = interval.service_ns
            row["work"] += 1
            row["measured"] += duration is not None
            row["modeled"] += estimate is not None
            if duration is None:
                if first_missing is None:
                    first_missing = reference.detail(interval)
                continue
            if estimate is not None:
                error = estimate - duration
                row["paired"] += 1
                row["signed_error_ns"] += error
                row["absolute_error_ns"] += abs(error)
        self.report["service_coverage"] = dict(coverage)
        self.report["observation_basis"] = dict(Counter(interval.basis for interval in reference.intervals))
        self.report["first_unmeasured_work"] = first_missing

    @staticmethod
    def first_drift(reference, plan, services, baseline, replayed):
        positions = reference.case.graph.positions
        first = reference.check_baseline(replayed)["first_difference"]
        if first is None:
            return None
        target = positions[first["rank"], first["event"]]
        tight = defaultdict(list)
        for source, consumer, lag in plan["dependencies"]:
            if (consumer == target or consumer >= len(positions)) and replayed[source] + lag == replayed[consumer]:
                tight[consumer].append(source)
        for (source, consumer, fixed), service in zip(plan["work"], services, strict=True):
            if (consumer == target or consumer >= len(positions)) and replayed[source] + service + fixed == replayed[consumer]:
                tight[consumer].append(source)

        # Auxiliary gates retain the nearest original prerequisite for attribution.
        pending = [target]
        visited = {target}
        identities = list(positions)
        predecessor = None
        for event in pending:
            predecessor = next((identities[source] for source in tight[event] if source < len(identities)), None)
            if predecessor is not None:
                break
            for source in tight[event]:
                if source not in visited:
                    pending.append(source)
                    visited.add(source)
        return {**first, "baseline_ns": baseline[target], "tight_predecessor": predecessor}

    def measured_baseline(self, reference, model, scratch, measured):
        if None in measured or "unqualified_host_service_partitions" in reference.gaps:
            self.report["baseline"] = {"status": "incomplete_measured_services", "missing_services": measured.count(None)}
            return None

        plan = reference.timeline()
        path = scratch / "replay.json"
        write_json(path, plan, replace=False)
        measured_steps = reference.replay_services(measured)
        services = scratch / "measured.json"
        write_json(services, {"measured": measured_steps}, replace=False)
        output = scratch / "measured-times.json"
        model.run_model("replay", output, path, services)
        times = json.loads(output.read_text())["event_times_ns"]["measured"]
        baseline = self.report["baseline"] = reference.check_baseline(times)
        publications = baseline["publications"] = reference.publications.compare(times)
        if baseline["first_difference"] is not None:
            baseline["status"] = "baseline_difference"
            baseline["local_unexplained_waits"] = reference.local_unexplained_waits(plan, measured_steps)
        elif publications["different_tokens"]:
            baseline["status"] = "baseline_difference"
        elif "incomplete_publication_boundaries" in reference.gaps:
            baseline["status"] = "incomplete_publication_boundaries"
        else:
            baseline["status"] = "measured_boundaries_matched"
            baseline["request_metrics"] = reference.publications.metrics(times)
        return plan, times

    def work_groups(self, reference, estimates, measured):
        if self.arguments.work_groups is None:
            return {}
        groups = json.loads(self.arguments.work_groups.read_text())
        allowed = {"kind", "scope", "site", "phase", "direction", "caller"}
        if not isinstance(groups, dict) or not groups:
            raise ValueError("work groups require one or more named selectors")
        for name, selector in groups.items():
            if (not isinstance(selector, dict) or "kind" not in selector or selector.keys() - allowed
                    or not all(isinstance(value, str) and value for value in selector.values())):
                raise ValueError(f"invalid work group selector: {name}")

        # Membership comes from the native graph even when a cost estimate is missing.
        members = {name: [] for name in groups}
        for index, interval in enumerate(reference.intervals):
            piece = interval.piece
            scope = reference.case.sources[interval.rank][0].graph["scopes"][piece["scope"]]
            copy = piece.get("copy", {})
            fields = {"kind": piece["kind"], "scope": scope["role"], "site": scope["site"], "phase": scope["work"].get("phase"),
                      "direction": copy.get("direction"), "caller": copy.get("caller")}
            for name, selector in groups.items():
                if all(fields[key] == value for key, value in selector.items()):
                    members[name].append(index)

        reports = self.report["work_groups"] = {}
        scenarios = {}
        for name, indices in members.items():
            paired = [i for i in indices if estimates[i] is not None and measured[i] is not None]
            errors = [estimates[i] - measured[i] for i in paired]
            row = reports[name] = {"selector": groups[name], "observed_work": len(indices), "paired_work": len(paired),
                                   "signed_service_error_ns": sum(errors), "absolute_service_error_ns": sum(map(abs, errors))}
            if not indices:
                row["status"] = "not_observed"
            elif len(paired) < len(indices):
                missing = next(i for i in indices if estimates[i] is None or measured[i] is None)
                row.update(status="incomplete_services", first_unpaired=reference.detail(reference.intervals[missing]))
            else:
                row["status"] = "services_paired"
                durations = measured.copy()
                for i in indices:
                    durations[i] = estimates[i]
                scenarios[f"only_{name}"] = durations
        return scenarios

    def scenarios(self, reference, model, scratch, estimates, measured, plan, times, scenarios):
        if None not in estimates:
            scenarios["modeled"] = estimates
            for kind in sorted(self.report["service_coverage"]):
                scenarios[f"restore_{kind}"] = [observed if interval.piece["kind"] == kind else estimate
                                                 for interval, estimate, observed in zip(reference.intervals, estimates, measured, strict=True)]
        if not scenarios:
            return
        scenarios = {name: reference.replay_services(durations) for name, durations in scenarios.items()}
        services = scratch / "scenarios.json"
        write_json(services, scenarios, replace=False)
        output = scratch / "scenario-times.json"
        model.run_model("replay", output, scratch / "replay.json", services)
        replayed = json.loads(output.read_text())["event_times_ns"]
        outcomes = {}
        for name, stamps in replayed.items():
            metrics = reference.publications.metrics(stamps)
            outcomes[name] = {"request_metrics": metrics, **compare_request_metrics(self.report["baseline"]["request_metrics"], metrics),
                              "first_timing_difference": self.first_drift(reference, plan, scenarios[name], times, stamps)}
        self.report["scenarios"] = outcomes

    def run(self):
        args = self.arguments
        self.output.mkdir(parents=True)
        recording = ReferenceRecording(args.capture, conditions={"service_reference": True})
        training = self.report["training"] = recording.training(costs=args.costs)
        if training["status"] != "separate":
            return self.finish(training["status"])
        checker = ProductionEvidence(args.reference, recording.source, self.output / "source-evidence", args.details)
        case, evidence = checker.inspect(retain_measurements=True)
        self.report["reference_status"] = evidence["native_case"]["status"]
        if checker.publish(evidence, require_joint=True) or case.groups.keys() != case.sources.keys():
            return self.finish("reference_unqualified")

        job = recording.inputs["job"]
        target = {field: job[field] for field in recording.target_fields}
        if difference := recording.input_difference(job, target) or case.input_difference(target):
            self.report["input_difference"] = difference
            return self.finish("input_mismatch")

        reference = FixedReference(case)
        self.report["reference_gaps"] = reference.gaps
        self.report["specialization_loads"] = reference.specialization_loads
        self.report["external_releases"] = [{"rank": rank, "event": event, "time_ns": stamp}
                                            for (rank, event), stamp in reference.entries.items()]
        self.report["idle_ingress_releases"] = [
            {"rank": rank, "predecessor": stopped.start, "successor": stopped.finish,
             "time_ns": wait.source.raw_ns - reference.origin_ns, "wait_enter_ns": stopped.observed_ns - reference.origin_ns}
            for (rank, stopped), wait in reference.input_releases.items()
        ]
        conditions = Counter()
        for row in reference.host_conditions.values():
            conditions.update(row)
        self.report["fixed_host_conditions"] = {"steps": len(reference.host_conditions), **conditions}
        self.report["fixed_host_fences"] = reference.fence_evidence
        self.report["fixed_device_dispatch"] = reference.device_dispatch
        model = ModelingProcess(args.binary, args.modeling_image, args.timeout)
        self.report["modeling_image"] = model.image
        with TemporaryDirectory(prefix=".fixed-", dir=self.output) as scratch:
            scratch = Path(scratch)
            measured = [interval.service_ns for interval in reference.intervals]
            replay = self.measured_baseline(reference, model, scratch, measured)

            # A same-run replay can match all measured endpoints even when changing service costs needs more evidence.
            estimated = self.estimate(reference, model, scratch)
            self.assess(reference, estimated)
            scoped = self.work_groups(reference, estimated, measured)
            if replay is None:
                return self.finish("reference_timing_incomplete")
            if self.report["baseline"]["status"] == "baseline_difference":
                return self.finish("baseline_difference")
            if reference.gaps:
                return self.finish("reference_timing_incomplete")
            if None in reference.specialization_loads.values():
                return self.finish("reference_specializations_unobserved")
            if any(reference.specialization_loads.values()):
                return self.finish("reference_loaded_specializations")
            self.scenarios(reference, model, scratch, estimated, measured, *replay, scoped)
            return self.finish("cost_gap" if None in estimated else "compared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="published native service-reference task")
    for name in ("reference", "costs", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    parser.add_argument("--timeout", type=float, required=True, help="cumulative modeling allowance in seconds")
    parser.add_argument("--details", action="store_true", help="retain all reference evidence findings")
    parser.add_argument("--work-groups", type=Path, help="replace only native work matching each named group and hold other services measured")
    raise SystemExit(0 if FixedPerformance(parser.parse_args()).run() else 1)


if __name__ == "__main__":
    main()
