"""Decode complete trace records and dispatch them to the owners of each evidence check."""

from collections import Counter

from prologue.verification.core.actions import GraphContents, QueryResults
from prologue.verification.core.protocol import IngressEvidence, PdPollingEvidence, PdPredictionEvidence
from prologue.verification.engine import CausalIndex, TraceReader

from .execution_inputs import ConsumedInputs
from .execution_times import ExecutionTimes
from .state_values import ActionResults, StateValueComparison
from .stream_queues import StreamSynchronization


class PredictionEvents:
    def __init__(self, case, comparison, calls):
        self.case, self.comparison, self.calls = case, comparison, calls
        self.event_ranks, self.completed, self.cross_rank_edges = {}, set(), set()
        self.ended, self.rejected = False, None
        self.native_reads = {}
        self.native_polling = PdPollingEvidence(self.case.protocol.admission_members)
        self.state_values = StateValueComparison(self.case.groups)
        self.action_results = ActionResults(self.case.groups)
        self.ingress = IngressEvidence(case.manifest["instances"], case.submissions)
        for instance in case.manifest["instances"]:
            for rank in instance["observed_roles"]:
                source, raw, identities = self.case.sources[rank]
                self.native_polling.native(raw, self.case.groups[rank])
                self.state_values.source(raw, self.case.groups[rank])
                self.action_results.source(raw, self.case.groups[rank], instance["mode"], source.source_ids)
                self.ingress.native(instance["id"], raw)
                for query in raw["state_queries"]:
                    if query["kind"] == "clock":
                        continue
                    index = source.source_ids[query["scope"]]
                    self.native_reads[self.case.groups[rank], index] = query["result"], identities[index]
        self.predicted_dependencies = {rank: CausalIndex() for rank in self.case.sources}
        self.condition_inputs = ConsumedInputs(case, self.ingress, comparison.realization,
            peer_handoffs=comparison.peer_handoffs)
        self.queries = QueryResults()
        self.execution_times = ExecutionTimes()
        self.stream_synchronization = StreamSynchronization()
        self.protocol = PdPredictionEvidence(self.case.protocol, self.case.readiness, self.ingress.origin_ns)

    def read(self, execution):
        trace = TraceReader(execution)
        for row in trace:
            self.dispatch(row)
        self.ended = trace.complete

    def dispatch(self, row):
        kind = row["kind"]
        self.condition_inputs.observe(row)
        if kind == "rejected_layout":
            self.rejected = self.comparison.rejected_action(row, self.condition_inputs)
            self.comparison.issues.counts["measured_layout_difference"] += 1
            return
        self.execution_times.observe(row)
        self.stream_synchronization.observe(row)
        self.case.cpu.observe(row)
        cpu_service = row["kind"] == "graph" and row["graph"]["scopes"][0]["role"] == "cpu_service"
        if not cpu_service:
            self.ingress.observe(row)
            self.case.effects.observe(row)
            self.case.connections.observe(row)
            self.state_values.observe(row)
            self.action_results.observe(row)
            query = self.queries.observe(row)
            self.protocol.observe(row, query, self.queries)
        self.boundary(row)
        if kind == "graph":
            self.graph(row, cpu_service)

    def boundary(self, row):
        if row["kind"] == "resources":
            self.case.bindings.initialized(row)
            self.case.effects.requests = [request["id"] for request in row["requests"]]
        elif row["kind"] == "notification_declared" or (row["kind"] == "native_protocol"
            and row["record"]["phase"] == "readiness"):
            group = row["identity"]["group"] if row["kind"] == "notification_declared" else row["group"]
            event = row["event"] if row["kind"] == "notification_declared" else row["record"]["event"]
            rank, = [rank for rank, owner in self.case.groups.items() if owner == group]
            if event in self.event_ranks:
                raise ValueError("external completion reuses an execution event identity")
            self.event_ranks[event] = rank
            self.predicted_dependencies[rank].add_events([event])
        elif row["kind"] == "resource":
            self.case.bindings.released(row)
        elif row["kind"] == "binding":
            self.case.bindings.bound(row)
        elif row["kind"] == "state_binding":
            self.case.bindings.lifetimes.state_bound(row)
        if row["kind"] == "edge":

            # CPU wire and wait edges have their own checked owner; every edge still enters the full graph below.
            if row["relation"] == "communication" and (row["source"],
                row["target"]) not in self.case.cpu.communication_edges:
                self.case.connections.communication_edges.add((row["source"], row["target"]))
            if row["relation"] == "signal":
                self.case.connections.signal_edges.add((row["source"], row["target"]))
            source_rank, target_rank = self.event_ranks[row["source"]], self.event_ranks[row["target"]]
            if source_rank == target_rank:
                self.predicted_dependencies[source_rank].connect(row["source"], row["target"], row["relation"])
            else:
                self.cross_rank_edges.add((row["source"], row["target"], row["relation"]))
        elif row["kind"] == "completion":
            event = row["event"]
            if event not in self.event_ranks or event in self.completed:
                raise ValueError("completion has an unknown or repeated event identity")
            self.completed.add(event)
            self.case.bindings.lifetimes.completed(event)

    def graph(self, row, cpu_service):
        graph = row["graph"]
        root = graph["scopes"][0]
        port = next(port for port in root["ports"] if port["name"] == "call.enter")
        rank = graph["resources"][graph["events"][port["event"]]["resource"]]["rank"]
        if any(resource["rank"] != rank for resource in graph["resources"]):
            raise ValueError("action contains resources owned by another rank")
        events = row["event_ids"]
        if len(events) != len(graph["events"]) or any(event in self.event_ranks for event in events):
            raise ValueError("execution has invalid invocation event identities")
        self.event_ranks.update(dict.fromkeys(events, rank))
        self.predicted_dependencies[rank].add_events(events)
        target = GraphContents(graph)
        conditioned_scope = self.condition_inputs.graph(self.case.groups[rank], row, target)
        if cpu_service:

            # Runtime protocol intervals are independently checked before joining action causality.
            return
        self.calls.observe(row, rank, port, target, conditioned_scope)

    def check_completion(self):
        if not self.ended:
            self.comparison.issues.add("incomplete_execution", {}, {})
        if len(self.completed) != len(self.event_ranks):
            self.comparison.issues.add("unfinished_events", {}, {"events": len(self.event_ranks) - len(self.completed)})
        # Dynamic waits and state hazards are recorded after graph insertion; compare only after reading the whole
        # execution.
        for graph in self.predicted_dependencies.values():
            graph.prepare()

    def cross_rank_coverage(self, cpu_scope):
        checked_cross_rank, unverified_cross_rank = Counter(), Counter()
        communication = set()
        if cpu_scope["complete"]:
            communication.update(self.case.cpu.communication_edges)
        if self.case.connections.counts["predicted_communication_protocol_complete"]:
            communication.update(self.case.connections.communication_edges)
        for source, target, relation in self.cross_rank_edges:

            # Edge ownership comes from completed protocol checks; cross-run action order is checked separately.
            covered = ((relation == "signal" and (source, target) in self.protocol.notification_edges)
                       or (relation == "communication" and (source, target) in communication))
            (checked_cross_rank if covered else unverified_cross_rank)[relation] += 1
        return checked_cross_rank, unverified_cross_rank

    def query_outcomes(self, pairs):
        query_counts = Counter()
        for key, (result, identity) in self.native_reads.items():
            if key not in pairs:
                query_counts["unpaired_native"] += 1
                continue
            if pairs[key] not in self.queries.results:
                query_counts["unreturned_predicted"] += 1
                continue
            actual = self.queries.results[pairs[key]]
            query_counts["compared"] += 1
            if actual != result:
                self.comparison.issues.add("query_outcome", identity,
                    {"native_scope": key[1], "predicted_event": pairs[key][1],
                                                        "native": result, "predicted": actual})
            else:
                query_counts["matched"] += 1
        query_counts["unpaired_predicted"] = len(self.queries.results.keys() - set(pairs.values()))
        return query_counts
