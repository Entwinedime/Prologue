"""Own the independently reconstructed native case shared by the verification routes."""

from collections import defaultdict
from functools import cached_property

from prologue.common.io import read_json
from prologue.common.observation import CapturedWorkers
from prologue.verification.core import RawCaseSource, RecordedSubmissions, ReferenceRecording
from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.core.adapters import device_coordinates
from prologue.verification.core.connections import Connections
from prologue.verification.core.cpu_protocol import CpuProtocolEvidence
from prologue.verification.core.graph_contents import GraphContents
from prologue.verification.core.pd_payload import PdPayloadEvidence
from prologue.verification.core.protocol import PdStateEvidence
from prologue.verification.core.resource_effects import ResourceEffectComparison
from prologue.verification.core.source_clock import ClockConstraints
from prologue.verification.core.state import BindingComparison
from prologue.verification.engine import CausalIndex, FindingLedger, TraceReader


class NativeCase:
    """Own the reference case built from qualified native records before any predicted actions are compared."""

    evidence_categories = frozenset(
        {"native_evidence", "binding_evidence", "cpu_protocol_evidence", "missing_native_identity"}
    )

    @classmethod
    def read(cls, capture, graphs):
        case = cls(RawCaseSource.read(capture), graphs)
        if not case.joint_execution:
            raise ValueError("a multi-worker target reference requires simultaneous execution on its physical topology")
        for rank in case.workers.observed:
            case.observe(rank, *TraceReader.worker(capture, graphs, rank))
        case.finish()
        return case

    def __init__(self, source: RawCaseSource, graphs):
        self.capture, self.manifest, self.graphs = source.directory, source.manifest, graphs
        self.workers = CapturedWorkers(self.manifest["instances"])
        self.ranks = self.workers.owners
        self.groups = {rank: f"{owner}/{rank}" for rank, owner in self.ranks.items()}
        self.observed_roles = set(self.workers.observed)
        self.joint_execution = self.workers.joint_execution

        self.sources, self.dependencies, self.clocks = {}, {}, {}
        self.logical = {}
        self.measurements = {}
        self.connections = Connections()
        self.cpu = CpuProtocolEvidence(self.capture, self.ranks)
        self.protocol = PdStateEvidence()
        self.control_declarations = defaultdict(set)
        self.findings = FindingLedger()
        self.observed_actions = 0

    @cached_property
    def sessions(self):
        return ReferenceRecording.sessions(self.capture, self.manifest["instances"])

    @cached_property
    def submissions(self):
        return RecordedSubmissions.read(self.capture, self.sessions)

    def input_difference(self, source):
        for _, raw, _ in self.sources.values():
            native = raw["environment"]["implementation"]
            if source["model"] != {
                "path": native["model"]["_name_or_path"],
                "dtype": native["dtype"].removeprefix("torch."),
            }:
                return "model"
        return ReferenceRecording.session_difference(source, self.manifest["instances"], self.sessions)

    def observe(self, rank, graph, raw, logical_coordinates=None):
        if rank not in self.observed_roles or rank in self.sources:
            raise ValueError("native worker observation is unexpected or repeated")
        if raw["environment"]["implementation"]["framework_revision"] != self.manifest["framework_revision"]:
            raise ValueError("native worker and capture disagree on their upstream framework revision")

        source = GraphContents(graph)
        source.validate_source(raw, rank)
        self.logical[rank] = set()
        if any(point["time_ns"] is None for point in graph["evidence"]["events"]):
            if logical_coordinates is None:
                logical_coordinates = device_coordinates(self.capture, self.graphs, rank, raw, graph)

            logical = {}
            for event, point in enumerate(graph["evidence"]["events"]):
                coordinate = graph["events"][event]["resource"], point["order"]
                if point["time_ns"] is None and coordinate in logical_coordinates:
                    logical[event] = logical_coordinates[coordinate]

            self.logical[rank] = set(logical)

            # A proven wait return can hand off a value produced by earlier work. The work itself still needs both
            # times.
            measured = {event for work in graph["work"] for event in (work["start"], work["finish"])}
            for scope in graph["scopes"]:
                for port in scope["ports"]:
                    if (port["use"] == "data" and (port["direction"] == "input" or logical.get(port["event"]) is None)
                            or port["use"] == "signal" and port["direction"] == "output"):
                        measured.add(port["event"])

                    if "rendezvous" in port:
                        measured.add(port["rendezvous"])

                for access in scope["kv_accesses"] + scope["slot_accesses"]:
                    measured.update((access["start"], access["finish"]))

            if invalid := measured & self.logical[rank]:
                raise ValueError(f"elided waits at rank {rank} own measured boundaries: {sorted(invalid)}")

        dependencies = CausalIndex.observed(graph)
        clock = ClockConstraints(graph, raw["device_clock"])
        measured_events = {event for event, point in enumerate(clock.points) if point is not None}

        # Clock inequalities are transitive, so this check needs no second graph.
        for source_event, target_event in dependencies.measured_dependencies(measured_events):
            clock.constrain(source_event, target_event)
        clock.finish()
        if clock.conflict:
            raise ValueError(
                f"native rank {rank} measurements have no common causal calibration: {clock.explain(graph)}"
            )

        self.clocks[rank] = clock
        identities = {}
        for index, scope in enumerate(graph["scopes"]):
            if not ActionIdentity.selected(index, scope):
                continue
            self.observed_actions += 1

            try:
                identity = source.record(index, self.groups[rank])
            except ValueError as error:
                identity = {"group": self.groups[rank], "scope": index, "action": scope["role"]}
                self.findings.add("missing_native_identity", identity, {"reason": str(error)})
            else:
                self.control_declarations[source.control_key(index, identity)].add(source.declared_control(index))

            identities[index] = identity

        if self.joint_execution:
            self.cpu.native_graph(rank, graph)
        self.protocol.observe(raw, self.groups[rank])
        self.dependencies[rank] = dependencies
        self.sources[rank] = source, raw, identities

        return source

    def completion_boundaries(self, rank, scope, *, peer_handoffs=False):
        source, _, _ = self.sources[rank]

        # The shared handoff uses named ports. Serial tails come from the physical implementation under comparison.
        for boundary in source.completion_boundaries(scope, named_only=peer_handoffs):
            if boundary["event"] not in self.logical[rank]:
                yield boundary

    def measured_service_pairs(self, rank, graph, raw, *, resources=None):
        if resources is None:
            resources = read_json(self.graphs / f"{rank}-facts.json")["resources"]
        payload = [{tuple(row) for row in lane["payload"]} for lane in resources]
        queries = {(raw["scopes"][read["scope"]]["thread"], read["begin_counter"], read["end_counter"])
                   for read in raw["state_queries"]}
        events = graph["evidence"]["events"]
        result = set()

        for piece in graph["work"]:
            lane = piece["resource"]
            resource = graph["resources"][lane]
            start, finish = piece["start"], piece["finish"]
            bounds = events[start]["order"], events[finish]["order"]

            # One actual task/copy, or an independently cut CPU read, has no inter-task gap to charge.
            if piece["kind"] in {"device", "transfer", "communication"} and bounds in payload[lane]:
                result.add((start, finish))
            elif piece["kind"] == "host" and resource["kind"] == "thread":
                thread = int(resource["identity"].split("/")[1])
                if (thread, *bounds) in queries:
                    result.add((start, finish))

        return result

    def finish(self):
        if self.sources.keys() != self.observed_roles:
            raise ValueError("native case omits observed workers")
        if not self.observed_actions:
            raise ValueError("native case contains no observed actions")

        self.controls = {key for key, declarations in self.control_declarations.items() if declarations == {True}}
        registrations = {}
        for path in sorted((self.capture / "kv-pools").glob("*.json")):
            registration = read_json(path)
            session = registration["registration"]["session"]
            if session in registrations:
                raise ValueError("native session identifies more than one registered pool")

            registrations[session] = registration

        registered_ranks = {session: value["parallel"]["rank"] for session, value in registrations.items()}

        # Registration covers initialized peers too; missing graphs never remove real PD endpoints.
        self.protocol.register(registrations, self.groups)
        self.histories = {}
        for rank, group in self.groups.items():
            path = self.capture / "resources" / f"rank-{rank}.json"
            if path.is_file():
                self.histories[group] = read_json(path)

        self.bindings = BindingComparison(
            self.histories, {session: self.groups[rank] for session, rank in registered_ranks.items()}
        )
        self.effects = ResourceEffectComparison()
        payload = PdPayloadEvidence(registrations, self.protocol.sessions, self.bindings.pools)

        for rank, (source, raw, identities) in self.sources.items():
            group = self.groups[rank]
            source.observed_pools(raw, registered_ranks, self.groups)
            payload.inspect(source, raw, group, self.dependencies[rank])
            self.connections.native(source, raw, rank, self.ranks[rank], self.dependencies[rank], identities)

            if group in self.histories:
                self.bindings.observed(source, raw, group, identities)
                self.effects.observed(self.histories[group], source.graph, group, raw)
            else:
                self.bindings.findings.add("binding_evidence", {"group": group},
                                           {"reason": "native resource history is absent"})

        self.findings.collect(payload.findings)
        self.payload_counts = dict(payload.counts)

        # These run-local rounds and links are also the source for later prediction comparison.
        self.connections.communication.native_communication(self.dependencies)
        self.pd_links = self.protocol.causal_dependencies(self.sources, self.groups)
        self.links = self.pd_links | {
            (source, target, "communication")
            for source, target in self.connections.communication.native_communication_edges
        }
        self.graph = self.cpu.native(self.dependencies, self.links)
        self.readiness = self.protocol.inspect()

    def report(self, details):
        ledger = FindingLedger(None if details else 1)
        ledger.collect(self.findings, self.connections.findings, self.bindings.findings, self.cpu.findings)
        missing_pd = {
            name: len(self.readiness[name])
            for name in (
                "missing_publications",
                "missing_reservation_receipts",
                "missing_reservation_updates",
                "missing_completion_updates",
                "missing_success_reads",
                "unobserved_transfer_peers",
            )
        }
        metadata_unknown = self.connections.counts.get("native_metadata_unobserved", 0)
        batch_unknown = any(self.connections.communication.batch_sync.counts.get("native_" + name, 0)
                            for name in ("unobserved", "unobserved_returns", "inactive_unverified"))
        cpu = self.cpu.report()
        incomplete = (any(missing_pd.values()) or metadata_unknown or batch_unknown
                      or not self.connections.counts["native_communication_protocol_complete"] or not cpu["complete"])
        outcome = ledger.outcome(self.evidence_categories, incomplete=incomplete, success="consistent")

        return {
            "status": outcome["status"],
            "workers": len(self.sources),
            "actions": self.observed_actions,
            "counts": ledger.counts,
            "findings": ledger.examples(self.evidence_categories),
            "model_communication": {
                name: self.connections.counts[name]
                for name in (
                    "communication_channels",
                    "communication_exchanges",
                    "native_communication_protocol_complete",
                    "native_peer_row_pairs",
                )
            },
            "model_links": len(self.connections.communication.native_communication_edges),
            "pd_links": len(self.pd_links),

            "cpu_protocol": cpu,
            "metadata_unobserved": metadata_unknown,
            "missing_pd_prerequisites": missing_pd,
            "physical_bindings": {
                "observed_request_bindings": len(self.bindings.native),
                "unconsumed_snapshots": self.bindings.unconsumed_snapshots,
            },
            "all_rank_events": None if self.graph is None else len(self.graph.next),
            "logical_wait_boundaries": sum(map(len, self.logical.values())),
            "pd_payload": self.payload_counts,

            "resource_reads": sum(value["count"] for value in self.effects.reads["native"].values()),

            "scope": (
                "Original worker declarations, empty logical resource pools, getter checkpoints, pool "
                "bindings, PD byte footprints/metadata and layer/K-V correspondence, enqueue slot ownership "
                "and retained page/row intervals, model rounds, CPU messages and PD prerequisites. No "
                "prediction, occurrence pairing, numerical payload verification or exact notification time."
            ),
        }
