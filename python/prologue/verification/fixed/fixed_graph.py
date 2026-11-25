"""Verify service costs without changing the observed serving execution."""

import json
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import write_json
from prologue.host.performance.requirements import CostRequirements
from prologue.host.runtime.docker import ModelingProcess
from prologue.verification.runner import VerificationRun

from ..core.metrics import compare_request_metrics
from ..core.source_evidence import ProductionEvidence
from .fixed_context import FixedHostContext
from .fixed_reference import FixedReference


class FixedPerformance(VerificationRun):
    recording_conditions = {"service_reference": True}

    def __init__(self, arguments):
        super().__init__(arguments)
        self.report = {"capture": str(arguments.capture.resolve()), "reference": str(arguments.reference.resolve()),
                       "cost_asset": str(arguments.costs.resolve())}

    def estimate(self, reference, model, scratch):
        estimates, domains, first_gaps = {}, {}, {}
        for rank, (contents, _, _) in reference.case.sources.items():
            output = scratch / f"{rank}-costs.json"
            model.run_model("work-costs", output, reference.case.graphs / f"{rank}-graph.json", self.arguments.costs)
            result = json.loads(output.read_text())
            estimates[rank] = result["work_ns"]
            domains[rank] = result["domain"]
            if len(estimates[rank]) != len(contents.graph["work"]):
                raise ValueError("fixed cost response changes the observed work set")
            if result["first_cost_gap"] is not None:
                first_gaps[rank] = result["first_cost_gap"]
        self.report["first_cost_gaps"] = first_gaps

        requirements = CostRequirements()
        points = {}
        for index, interval in enumerate(reference.intervals):
            if external := reference.external_cost(interval):
                rank, identity, features = external
                points[index] = requirements.external(
                    "fixed_protocol", domains[rank], identity, interval.kind, features
                )
        costs = {}
        if points:
            asset = json.loads(self.arguments.costs.read_text())["domains"]
            costs = requirements.query(model, asset, scratch, nullcontext, None)

        result, first_protocol_gap = [], None
        for index, interval in enumerate(reference.intervals):
            if index in points:
                point = points[index]
                status, value = costs[point]
                if value is None and first_protocol_gap is None:
                    first_protocol_gap = {**reference.detail(interval), "status": status, "features": point.features()}
                result.append(value)
            else:
                result.append(estimates[interval.rank][interval.index])
        if first_protocol_gap is not None:
            self.report["first_protocol_cost_gap"] = first_protocol_gap
        return result

    def assess(self, reference, estimates):
        coverage = defaultdict(lambda: {"work": 0, "measured": 0, "modeled": 0, "paired": 0,
                                        "signed_error_ns": 0, "absolute_error_ns": 0})
        first_missing = None
        for interval, estimate in zip(reference.intervals, estimates, strict=True):
            row = coverage[interval.kind]
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
            if (consumer == target or consumer >= len(positions)) and replayed[source] + service + fixed == replayed[
                consumer
            ]:
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
        if None in measured or "unqualified_work_service_partitions" in reference.gaps:
            self.report["baseline"] = {
                "status": "incomplete_measured_services",
                "missing_services": measured.count(None),
            }
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
        transport = baseline["cpu_transport"] = reference.transport_baseline(times)
        inputs = baseline["active_input_reads"] = reference.compare_active_input_reads(times, measured=True)
        if baseline["first_difference"] is not None:
            baseline["status"] = "baseline_difference"
            baseline["local_unexplained_waits"] = reference.local_unexplained_waits(plan, measured_steps)
        elif publications["different_tokens"] or transport["different"] or inputs["different"]:
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
            fields = reference.work_fields(interval)
            for name, selector in groups.items():
                if all(fields[key] == value for key, value in selector.items()):
                    members[name].append(index)

        reports = self.report["work_groups"] = {}
        scenarios = {}
        for name, indices in members.items():
            paired = [i for i in indices if estimates[i] is not None and measured[i] is not None]
            errors = [estimates[i] - measured[i] for i in paired]
            row = reports[name] = {
                "selector": groups[name],
                "observed_work": len(indices),
                "paired_work": len(paired),
                "signed_service_error_ns": sum(errors),
                "absolute_service_error_ns": sum(map(abs, errors)),
            }
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
        if not reference.gaps and None not in estimates:
            scenarios["modeled"] = estimates
            for kind in sorted(self.report["service_coverage"]):
                scenarios[f"restore_{kind}"] = [
                    observed if interval.kind == kind else estimate
                    for interval, estimate, observed in zip(reference.intervals, estimates, measured, strict=True)
                ]
        if not scenarios:
            return True
        scenarios = {name: reference.replay_services(durations) for name, durations in scenarios.items()}
        services = scratch / "scenarios.json"
        write_json(services, scenarios, replace=False)
        output = scratch / "scenario-times.json"
        model.run_model("replay", output, scratch / "replay.json", services)
        replayed = json.loads(output.read_text())["event_times_ns"]
        outcomes = {}
        for name, stamps in replayed.items():
            inputs = reference.compare_active_input_reads(stamps)
            outcome = outcomes[name] = {
                "active_input_reads": inputs,
                "first_timing_difference": self.first_drift(reference, plan, scenarios[name], times, stamps),
            }
            if inputs["different"]:
                outcome["status"] = "input_readiness_unverified"
                continue

            metrics = reference.publications.metrics(stamps)
            outcome.update(
                request_metrics=metrics, **compare_request_metrics(self.report["baseline"]["request_metrics"], metrics)
            )
        self.report["scenarios"] = outcomes
        return outcomes.get("modeled", {}).get("status") != "input_readiness_unverified"

    def verify(self):
        args = self.arguments
        recording = self.recording
        if status := self.training_gap(costs=args.costs):
            return self.finish(status)
        checker = ProductionEvidence(args.reference, recording.source, self.output / "source-evidence", args.details)
        case, evidence = checker.inspect(retain_measurements=FixedHostContext.retain)
        self.report["reference_status"] = evidence["native_case"]["status"]
        if checker.publish(evidence, require_joint=True) or case.groups.keys() != case.sources.keys():
            return self.finish("reference_unqualified")

        job = recording.inputs["job"]
        target = {field: job[field] for field in recording.target_fields}
        if self.input_mismatch(target) or self.input_mismatch(target, case=case):
            return self.finish("input_mismatch")

        reference = FixedReference(case)
        self.report["reference_gaps"] = reference.gaps
        if reference.cpu_calls:
            self.report["cpu_protocol"] = {
                "host_calls": len(reference.cpu_calls), "measured_endpoints": len(reference.cpu_points),
                "measured_transport_endpoints": len(reference.cpu_transfers),
                "initial_transport_conditions": len(reference.cpu_initial_readiness),
            }
        self.report["specialization_loads"] = reference.specialization_loads
        self.report["external_releases"] = [{"rank": rank, "event": event, "time_ns": stamp}
                                            for (rank, event), stamp in reference.entries.items()]
        self.report["idle_ingress_releases"] = [
            {
                "rank": rank,
                "predecessor": stopped.start,
                "successor": stopped.finish,
                "time_ns": wait.source.raw_ns - reference.origin_ns,
                "wait_enter_ns": stopped.observed_ns - reference.origin_ns,
            }
            for (rank, stopped), wait in reference.input_releases.items()
        ]
        conditions = Counter()
        for row in reference.host_conditions.values():
            conditions.update(row)
        self.report["fixed_host_conditions"] = {"steps": len(reference.host_conditions), **conditions}
        if reference.cross_worker_wakes:
            self.report["cross_worker_wakes"] = reference.cross_worker_wakes
        if reference.post_graph_host_wakes:
            self.report["post_graph_host_wakes"] = reference.post_graph_host_wakes
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
            if reference.gaps.keys() - {"unmeasured_cpu_transport"}:
                return self.finish("reference_timing_incomplete")
            if None in reference.specialization_loads.values():
                return self.finish("reference_specializations_unobserved")
            if any(reference.specialization_loads.values()):
                return self.finish("reference_loaded_specializations")
            if reference.gaps and scoped:
                self.report["scoped_scenario_basis"] = "hold_observed_host_wakes_and_wire_dependencies"
            input_ready = self.scenarios(reference, model, scratch, estimated, measured, *replay, scoped)
            if reference.gaps or not input_ready:
                return self.finish("reference_timing_incomplete")
            return self.finish("cost_gap" if None in estimated else "compared")
