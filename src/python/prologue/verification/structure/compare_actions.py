"""Compare action contents, causal boundaries and data/communication connections with a native case."""

from prologue.common.components import contributions
from prologue.verification.core.action_order import ActionOrderComparison
from prologue.verification.core.cpu_payloads import compare_wire_payloads
from prologue.verification.core.native_case import NativeCase
from prologue.verification.engine import FindingLedger

from . import checks
from .action_report import COMPARISON_SCOPE, write_action_report
from .invocations import InvocationComparison
from .local_graphs import LocalActionComparison
from .trace_events import PredictionEvents


class ActionComparison:
    """Pair native and predicted actions by their observed work identity before comparing contents and causal
    relationships.
    """

    evidence_categories = NativeCase.evidence_categories | frozenset(
        {
            "state_binding_evidence",
            "ordering_evidence",
            "input_read_evidence",
            "stream_evidence",
            "missing_predicted_identity",
            "ambiguous_native_identity",
            "ambiguous_predicted_identity",
            "unpaired_native_context",
            "unpaired_predicted_context",
            "incomplete_execution",
            "unfinished_events",
            "native_tail_unexecuted",
        }
    )

    scope_limits = tuple([
                "exact PD notification arrival times",
                "unnamed internal event topology",
                "private intra-call resource mutations and tensor hazards; native in-flight resource reuse outside "
                "observed cases",
                "PD unused padding and unsupported multimodal/logprob/speculative values; controller output values "
                "outside recorded progress and publication fields",
            ])

    def __init__(self, reference, execution, output, details, *, realization=None, peer_handoffs=False):
        self.reference, self.execution, self.output, self.details = reference, execution, output, details
        self.realization, self.peer_handoffs = realization, peer_handoffs
        self.capture, self.graphs = reference.capture, reference.graphs
        self.issues = FindingLedger(None if details else 1)
        registered = tuple(contributions(checks, "checks"))
        self.checks = {check.name: check for check in registered}

        if len(self.checks) != len(registered):
            raise ValueError("verification check names must be unique")
        self.evidence_categories = self.evidence_categories.union(
            *(check.evidence_gaps for check in self.checks.values())
        )
        self.check_results = {}

    def rejected_action(self, row, conditions):
        group, step, graph = row["group"], row["step"], row["graph"]
        root = graph["scopes"][0]
        record = conditions.matched_invocation(group, root)

        if step + 1 != conditions.positions[group] or record["operation"] != "action":
            raise ValueError("rejected graph differs from the witnessed native action position")

        # This graph never entered the timeline. Check its contents and intrinsic dependencies separately from the
        # executed prefix.
        rank, scope = conditions.ranks[group], record["source"]["scope"]
        case = self.reference
        source, _, identities = case.sources[rank]
        observed = source.graph["scopes"][scope]
        comparison = LocalActionComparison()
        comparison.compare(case, rank, scope, graph)

        return {
            "group": group,
            "step": step,
            "site": root["site"],
            "action": identities[scope],
            "native": {
                "worker": str(case.capture / f"rank-{rank}.json"),
                "scopes": observed["bindings"]["source_scopes"],
                "graph": str(case.graphs / f"{rank}-graph.json"),
                "scope": scope,
            },
            "local_checks": dict(comparison.counts),
            "examples": comparison.findings,
        }

    def join_graphs(self, trace, calls, ordering):
        case = self.reference
        case.connections.compare_signals(calls.pairs)
        case.connections.communication.compare_predicted_communication(trace.predicted_dependencies)
        case.connections.communication.compare_communication(calls.pairs)
        predicted_links = {
            ((trace.event_ranks[source], source), (trace.event_ranks[target], target), relation)
            for source, target, relation in trace.cross_rank_edges
        }
        actual_graph = case.graph
        predicted_graph = case.cpu.prediction(trace.predicted_dependencies, predicted_links, complete=trace.ended)
        case.connections.check_predicted_readiness(predicted_graph, trace.event_ranks)
        case.connections.compare_values(predicted_graph, calls.pairs)
        cpu_scope = case.cpu.report()
        wire, wire_differences = compare_wire_payloads(
            case.cpu.messages["native"], case.cpu.messages["predicted"], trace.condition_inputs.cpu_messages
        )
        cpu_scope["wire_payloads"] = wire

        for detail in wire_differences:
            self.issues.add("cpu_wire_payload", {}, detail)
        order_ready = (
            case.connections.counts["native_communication_protocol_complete"]
            and case.connections.counts["predicted_communication_protocol_complete"]
            and cpu_scope["complete"]
        )

        if order_ready:
            calls.compare_boundaries(actual_graph, predicted_graph, ordering)
        else:
            self.issues.add(
                "ordering_evidence",
                {},
                {
                    "reason": "all-rank causal comparison needs complete model communication and CPU protocols",
                    "native_model_protocol": case.connections.counts["native_communication_protocol_complete"],
                    "predicted_model_protocol": case.connections.counts["predicted_communication_protocol_complete"],
                    "cpu_protocol": cpu_scope["complete"],
                },
            )

        case.bindings.compare(calls.pairs)
        case.bindings.lifetimes.compare(predicted_graph, trace.event_ranks)
        return cpu_scope

    def check_effects(self, trace, calls, ordering, checked_prefix):
        case = self.reference
        stream_scope = trace.stream_synchronization.finish()
        self.issues.collect(
            case.connections.findings,
            case.bindings.findings,
            case.bindings.lifetimes.findings,
            ordering.findings,
            case.cpu.findings,
            trace.stream_synchronization.findings,
            case.effects.compare(complete=trace.ended),
            trace.state_values.compare(case.protocol),
            trace.action_results.compare(calls.pairs),
            trace.native_polling.findings,
            trace.protocol.polling.findings,
        )

        input_delivery = trace.ingress.compare(
            trace.condition_inputs.input_pairs if checked_prefix else None, complete=trace.ended
        )
        self.issues.collect(trace.ingress.findings)

        for instance, result in input_delivery.items():
            if result["status"] != "matched":
                category = "native_evidence" if result["status"] == "missing_native_evidence" else "request_ingress"
                self.issues.add(category, {"instance": instance}, result)

            if result.get("input_reads", {}).get("status") == "different":
                self.issues.add("input_read", {"instance": instance}, result["input_reads"])

        for name, check in self.checks.items():
            findings = tuple(check.findings(case, trace))
            self.issues.collect(findings)
            self.check_results[name] = len(findings)

        return stream_scope, input_delivery

    def run(self):
        case = self.reference
        calls = InvocationComparison(case, self.issues.add, self.details)
        trace = PredictionEvents(case, self, calls)
        self.issues.collect(case.findings)
        native_evidence = case.report(self.details)

        trace.read(self.execution)
        condition_scope = trace.condition_inputs.finish()
        checked_prefix = condition_scope["verified_prefix"]
        calls.check_counts(trace.ended, checked_prefix)
        trace.check_completion()
        calls.pair()

        ordering = ActionOrderComparison(calls.keys, self.details)
        cpu_scope = self.join_graphs(trace, calls, ordering)
        stream_scope, input_delivery = self.check_effects(trace, calls, ordering, checked_prefix)
        recorded_queries = trace.queries.finish(complete=trace.ended)
        checked_cross_rank, unverified_cross_rank = trace.cross_rank_coverage(cpu_scope)
        query_counts = trace.query_outcomes(calls.pairs)
        report = {
            "native_case": native_evidence,
            "condition_inputs": condition_scope,
            "cpu_protocol": cpu_scope,
            "stream_synchronization": stream_scope,
            "request_ingress": input_delivery,
            "query_results_recorded": recorded_queries,
            "query_outcomes": query_counts,
        }
        report.update(self.summarize(trace, calls, ordering, report, checked_cross_rank, unverified_cross_rank))

        if self.details:
            report["registered_checks"] = self.check_results
        write_action_report(self.output, report)
        return report

    def qualification(self, trace, ordering, report, unpaired_calls, unverified_cross_rank):
        case = self.reference
        cpu_scope = report["cpu_protocol"]
        wire = cpu_scope["wire_payloads"]
        input_incomplete = any(
            result.get("input_reads", {}).get("status") != "matched" for result in report["request_ingress"].values()
        )
        metadata_incomplete = bool(case.connections.communication.metadata_returns) or any(
            case.connections.counts.get(f"{origin}_metadata_unobserved", 0) for origin in ("native", "predicted")
        )

        case.connections.communication.batch_sync.counts["predicted_unobserved_returns"] = sum(
            case.connections.communication.batch_sync.terminal(part["work"].get("work", {}))
            for part in case.connections.communication.action_returns.values()
        )
        batch_sync_incomplete = any(
            case.connections.communication.batch_sync.counts.get(f"{origin}_{reason}", 0)
            for origin in ("native", "predicted")
            for reason in ("unobserved", "unobserved_returns", "inactive_unverified")
        )
        admission_incomplete = any(
            check["unobserved"]
            for checker in (trace.native_polling, trace.protocol.polling)
            for check in (checker.operand_counts(), checker.commit_counts())
        )

        payload_incomplete = any(
            trace.state_values.counts.get(f"{origin}_unpaired", 0) for origin in ("native", "predicted")
        )
        connections_incomplete = (
            not case.connections.counts["data_evidence_complete"]
            or not case.connections.counts["communication_evidence_complete"]
            or case.connections.counts["data_inputs_unpaired"]
            or case.connections.counts["signal_inputs_unpaired"]
        )
        bindings_incomplete = (
            case.bindings.counts["request_bindings_unpaired"]
            or case.bindings.counts["predicted_request_bindings_unpaired"]
        )
        queries_incomplete = (
            report["query_outcomes"]["unpaired_native"]
            or report["query_outcomes"]["unpaired_predicted"]
            or report["query_outcomes"]["unreturned_predicted"]
        )

        incomplete = (
            payload_incomplete
            or any(unpaired_calls.values())
            or input_incomplete
            or metadata_incomplete
            or admission_incomplete
            or connections_incomplete
            or batch_sync_incomplete
            or bindings_incomplete
            or queries_incomplete
            or bool(case.effects.counts["state_slots"]["unobserved_native"])
            or bool(unverified_cross_rank)
            or not cpu_scope["complete"]
            or wire["unresolved"]
            or not ordering.qualified
            or report["condition_inputs"]["status"] == "partial"
        )

        return {
            **self.issues.outcome(self.evidence_categories, incomplete=incomplete, success="actions_matched"),

            "not_checked": [
                detail
                for missing, detail in (
                    (queries_incomplete, "unpaired query outcomes"),
                    (input_incomplete, "unpaired external input reads"),
                    (
                        cpu_scope.get("native_messages", 0) and not wire["witnessed"],
                        "CPU payloads without paired post witnesses",
                    ),
                    (wire["unresolved"], "witnessed CPU payloads without both posts in each run"),
                    (
                        bool(unverified_cross_rank),
                        "cross-rank causality outside verified communication and notification protocols",
                    ),
                    (
                        any(unpaired_calls.values()),
                        "per-occurrence correspondence of calls without an independently checked witness",
                    ),
                    (
                        bool(trace.condition_inputs.counts["bound_invocations"]),
                        "autonomous prediction of explicitly supplied native invocation bindings",
                    ),

                    (self.peer_handoffs, "timing outside the matched layouts and shared handoff boundaries"),
                )
                if missing
            ]

            + list(self.scope_limits),
        }

    def summarize(self, trace, calls, ordering, report, checked_cross_rank, unverified_cross_rank):
        case = self.reference
        unpaired = calls.unpaired()
        return {
            **self.qualification(trace, ordering, report, unpaired, unverified_cross_rank),
            **calls.report(),
            "inputs": {
                "capture": str(self.capture.resolve()),
                "graphs": str(self.graphs.resolve()),
                "execution": str(self.execution.resolve()),
            },

            "first_rejected_action": trace.rejected,
            "unpaired_calls": unpaired,
            "counts": dict(self.issues.counts),
            "findings": self.issues.examples(self.evidence_categories),
            "connections": case.connections.counts,
            "batch_synchronization": dict(case.connections.communication.batch_sync.counts),

            "action_order": {
                **ordering.counts,
                "graph": "all_ranks_model_cpu_pd_and_state",
                "coverage": "checked_scope" if ordering.qualified else "unqualified",
                "scope": (
                    "Project full all-rank paths onto already-paired named events, preserving aliases and paths "
                    "through unpaired intermediates. Compare each order's minimal relations to the opposite "
                    "reachable order; both directions together retain strict order equality. A differing relation "
                    "is not a proven core defect, and an opposite reverse path forbids simply adding that edge. No "
                    "new occurrence pairing, timing or state-safety qualification."
                ),
            },

            "repeated_call_order": {
                **ordering.repeated_counts,
                "scope": (
                    "Sets of all-rank causal positions relative to uniquely paired call entries/returns in the same "
                    "joined graphs. Counts, payloads, effects, cross-rank rounds and timing are not paired; no "
                    "source occurrence is selected by matching its position."
                ),
            },

            "physical_bindings": case.bindings.counts,
            "state_lifetimes": case.bindings.lifetimes.report(),
            "resource_effects": case.effects.counts,
            "state_values": trace.state_values.counts,
            "action_results": dict(trace.action_results.counts),
            "execution_times": trace.execution_times.finish(complete=trace.ended),
            "completion_reads_checked": trace.queries.completion_reads,
            "completion_notifications_checked": trace.queries.notifications,

            "pd_read_prerequisites": trace.protocol.finish(),
            "native_protocol_readiness": case.readiness,
            "native_polling_operands": trace.native_polling.operand_counts(),
            "native_admission_decisions": trace.native_polling.commit_counts(),
            "cross_rank_dependencies": {
                "protocol_checked_edges": dict(checked_cross_rank),
                "unverified_edges": dict(unverified_cross_rank),
                "scope": (
                    "Run-local edges owned by complete model/CPU protocols or independently checked PD "
                    "notifications. Named-action causality and witnessed CPU payloads have their own cross-run "
                    "checks."
                ),
            },

            "scope": COMPARISON_SCOPE,
        }
