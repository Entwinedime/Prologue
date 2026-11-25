"""Build every native worker from its recording and qualify the projected execution graph."""

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from functools import cached_property
from heapq import heappop, heappush
from pathlib import Path

from prologue.common.io import read_json
from prologue.common.observation import CapturedWorkers
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.libraries.release import asset_path
from prologue.verification.core import RecordedSubmissions, ReferenceRecording
from prologue.verification.core.actions import ActionIdentity, ActionScopes, GraphContents
from prologue.verification.core.adapters import device_coordinates, device_effects, stream_declarations
from prologue.verification.core.connections import Connections, CpuProtocolEvidence
from prologue.verification.core.protocol import PdPayloadEvidence, PdStateEvidence
from prologue.verification.core.source_clock import ClockConstraints
from prologue.verification.core.state import BindingComparison, ResourceEffectComparison
from prologue.verification.engine import CausalIndex, FindingLedger, ReportBuilder, TraceReader


class NativeCase:
    evidence_categories = frozenset(
        {"native_evidence", "binding_evidence", "cpu_protocol_evidence", "missing_native_identity"}
    )

    @classmethod
    def read(cls, capture, graphs):
        manifest = read_json(capture / "capture.json")
        case = cls(capture, manifest, graphs)
        if not case.joint_execution:
            raise ValueError("a multi-worker target reference requires simultaneous execution on its physical topology")
        for rank in case.workers.observed:
            case.observe(rank, *TraceReader.worker(capture, graphs, rank))
        case.finish()
        return case

    def __init__(self, capture, manifest, graphs):
        self.capture, self.manifest, self.graphs = capture, manifest, graphs
        self.workers = CapturedWorkers(manifest["instances"])
        self.ranks = self.workers.owners
        self.groups = {rank: f"{owner}/{rank}" for rank, owner in self.ranks.items()}
        self.observed_roles = set(self.workers.observed)
        self.joint_execution = self.workers.joint_execution

        self.sources, self.dependencies, self.clocks = {}, {}, {}
        self.logical = {}
        self.measurements = {}
        self.connections = Connections()
        self.cpu = CpuProtocolEvidence(capture, self.ranks)
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
        self.connections.native_communication(self.dependencies)
        self.pd_links = self.protocol.causal_dependencies(self.sources, self.groups)
        self.links = self.pd_links | {
            (source, target, "communication") for source, target in self.connections.native_communication_edges
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
        batch_unknown = any(self.connections.batch_sync.counts.get("native_" + name, 0)
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
            "model_links": len(self.connections.native_communication_edges),
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


class SourceEvidence:
    def __init__(self, facts, graph, clock, logical, details=False):
        self.facts, self.graph = facts, graph
        self.evidence = graph["evidence"]
        self.counts = Counter()
        self.findings = FindingLedger(None if details else 3)
        external = {event for event, point in enumerate(self.evidence["events"])
                    if point["time_ns"] is None and "protocol_operand" in facts["spans"][point["span"]]
                    and graph["resources"][graph["events"][event]["resource"]]["kind"] == "service"}
        self.unmeasured_boundaries = logical | external
        self.maximum_inversion_ns = 0
        self.coverage = {}
        self.clock, self.logical = clock, logical

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def coordinate(self, anchor):
        span = self.facts["spans"][anchor["span"]]
        self.require(span["begin"] <= anchor["order"] <= span["end"], f"anchor outside span: {anchor}")
        resource = span["resource"]

        # Serial boundaries can be shared by adjacent spans; concurrent services retain their own span identity.
        return resource, anchor["order"], None if self.facts["resources"][resource]["serial"] else anchor["span"]

    def validate_mapping(self):
        facts, graph, evidence = self.facts, self.graph, self.evidence
        self.require(len(graph["events"]) == len(evidence["events"]), "event evidence count differs")
        self.require(len(graph["work"]) == len(evidence["work_spans"]), "work evidence count differs")
        self.require(len(graph["scopes"]) == len(facts["scopes"]), "source scope count differs")
        self.require(graph["environment"] == facts["environment"], "source environment differs")
        self.require(graph["resources"] == [
            {key: value for key, value in resource.items() if key not in {"observed", "payload"}}
            for resource in facts["resources"]
        ], "source resource descriptors differ")
        measurements = facts["measurements"]
        self.require(
            all(evidence[key] == measurements[key] for key in ("clock", "window_ns")),
            "measurement clock/window differs",
        )
        times = {(item["resource"], item["order"]): item["time_ns"] for item in measurements["coordinates"]}
        coordinates = {}
        for index, (event, anchor) in enumerate(zip(graph["events"], evidence["events"], strict=True)):
            key = self.coordinate(anchor)
            self.require(key not in coordinates, f"duplicate projected coordinate: {key}")
            coordinates[key] = index
            self.require(event["resource"] == key[0], f"event {index} changes source resource")
            self.require(anchor["time_ns"] == times[key[:2]], f"event {index} changes source measurement")

        required = set()

        def resolve(anchor):
            coordinate = self.coordinate(anchor)
            required.add(coordinate)
            return coordinates[coordinate]

        # Source endpoints are coordinates; after resolving them both declarations share one comparison.
        for index, (native, projected) in enumerate(zip(facts["scopes"], graph["scopes"], strict=True)):
            expected = self.scope_contract(native, resolve)
            actual = self.scope_contract(projected)
            if expected != actual:
                changed = sorted(key for key in expected.keys() | actual.keys()
                                 if key not in expected or key not in actual or expected[key] != actual[key])
                raise ValueError(f"scope {index} changes source fields: {changed}")

        edges = {(edge["source"], edge["target"], edge["kind"]) for edge in graph["edges"]}
        native_edges = set()
        for edge in facts["dependencies"]:
            source, target = resolve(edge["source"]), resolve(edge["target"])
            if source != target:
                native_edges.add((source, target, edge["kind"]))
        orders = self.validate_coverage(coordinates, required)
        self.require(edges == native_edges | orders, "projected dependencies omit or add source/serial relations")
        self.require(len(edges) == len(graph["edges"]), "projected dependencies repeat")

    @staticmethod
    def scope_contract(scope, resolve=None):
        declaration = scope | ActionScopes.declaration_expressions(scope)
        ports = []
        for observed in scope["ports"]:
            port = observed.copy()
            if resolve:
                port["event"] = resolve(port.pop("at"))
                if "rendezvous" in port:
                    port["rendezvous"] = resolve(port["rendezvous"])
                if "wait" in port:
                    wait = port["wait"]
                    port["wait"] = {"resource": wait["resource"], "begin": resolve(wait["begin"])}
            if "tensor" in port:
                tensor = port["tensor"]
                port["tensor"] = tensor | {"shape_rule": ActionScopes.expression_list_key(tensor.get("shape_rule"))}
            ports.append(port)
        declaration["ports"] = ports
        for field in ("kv_accesses", "slot_accesses"):
            values = []
            for access in scope[field]:
                value = access | {"rule": ActionScopes.expression_key(access.get("rule"))}
                if resolve:
                    value.update({side: resolve(access[side]) for side in ("start", "finish")})
                values.append(value)
            declaration[field] = values
        return declaration

    def validate_coverage(self, coordinates, required):
        ancestors = []
        for index, scope in enumerate(self.facts["scopes"]):
            parent = scope["parent"]
            self.require((parent is None and index == 0) or (parent is not None and 0 <= parent < index),
                         f"scope {index} has an invalid source parent")
            ancestors.append({index} | (ancestors[parent] if parent is not None else set()))

        spans, boundaries, touching = defaultdict(list), defaultdict(list), defaultdict(set)
        for index, span in enumerate(self.facts["spans"]):
            resource = self.facts["resources"][span["resource"]]
            owner = None if resource["serial"] else index
            spans[span["resource"], owner].append((index, span))
            for position in (span["begin"], span["end"]):
                required.add((span["resource"], position, owner))
                touching[span["resource"], position, owner].add(span["scope"])
        self.require(required == coordinates.keys(), "projected boundaries omit or add source coordinates")

        for (resource, position, owner), event in coordinates.items():
            boundaries[resource, owner].append((position, event))

        expected, orders = Counter(), set()
        for key, points in boundaries.items():
            resource = self.facts["resources"][key[0]]
            payload = resource["payload"]
            payload_ends = [end for _, end in payload]
            pending = iter(sorted(spans[key], key=lambda item: item[1]["begin"]))
            candidate = next(pending, None)
            active = []
            points.sort()
            preceding = None
            for position, (begin, start) in enumerate(points):
                while candidate is not None and candidate[1]["begin"] <= begin:
                    index, span = candidate

                    # Innermost span owns work; a nested wait suppresses its parent's service cost.
                    heappush(active, (span["end"], -span["begin"], -span["scope"], span["kind"] != "wait", -index))
                    candidate = next(pending, None)
                while active and active[0][0] <= begin:
                    heappop(active)
                index = -active[0][-1] if active else None
                span = self.facts["spans"][index] if index is not None else None

                # A shared boundary belongs to the common ancestor of both sides and all touching spans.
                owners = touching[key[0], begin, key[1]].copy()
                if preceding is not None:
                    owners.add(preceding)
                if span is not None:
                    owners.add(span["scope"])
                common = set.intersection(*(ancestors[owner] for owner in owners))
                self.require(
                    self.graph["events"][start]["scope"] == max(common), f"event {start} changes source semantic owner"
                )
                preceding = span["scope"] if span is not None else None
                if position + 1 == len(points):
                    continue
                end, finish = points[position + 1]
                has_work = False
                if span is not None:
                    self.require(end <= span["end"], f"resource {key} skips an active span boundary")
                    payload_index = bisect_right(payload_ends, begin)
                    has_payload = payload_index < len(payload) and payload[payload_index][0] < end
                    has_work = span["kind"] not in {"wait", "boundary"} and (
                        span["kind"] == "host" or has_payload or "protocol_operand" in span)
                if has_work:
                    copy = span.get("copy") if (begin, end) == (span["begin"], span["end"]) else None
                    expected[index, span["scope"], key[0], span["kind"], start, finish,
                             json.dumps(copy, sort_keys=True) if copy else None] += 1
                else:
                    orders.add((start, finish, "order"))

        # Compare the whole partition, not just the pieces that survived projection.
        actual = Counter((span, work["scope"], work["resource"], work["kind"], work["start"], work["finish"],
                          json.dumps(work["copy"], sort_keys=True) if "copy" in work else None)
                         for span, work in zip(self.evidence["work_spans"], self.graph["work"], strict=True))
        self.require(
            actual == expected,
            f"source work coverage differs: {sum((expected - actual).values())} missing, "
            f"{sum((actual - expected).values())} extra",
        )
        self.coverage = {"native_spans": len(self.facts["spans"]), "work_intervals": sum(expected.values()),
                         "ordering_intervals": len(orders)}
        return orders

    def timing(self, kind, index, source, target):
        begin = self.evidence["events"][source]["time_ns"]
        end = self.evidence["events"][target]["time_ns"]
        if begin is None or end is None:
            missing = {event for event, time in ((source, begin), (target, end)) if time is None}
            if missing <= self.logical:
                self.counts["logical_condition_relations"] += 1
                return
            if missing <= self.unmeasured_boundaries:
                self.counts["external_transport_relations"] += 1
                return
            issue, difference = "unknown_time", None
        elif begin > end:
            issue, difference = "time_inversion", begin - end
            self.maximum_inversion_ns = max(self.maximum_inversion_ns, difference)
        else:
            return

        self.findings.add(issue, {"kind": kind, "index": index},
                          {"source": source, "target": target, "source_ns": begin, "target_ns": end,
                           "difference_ns": difference})


    def inspect(self):
        self.validate_mapping()

        # The case clock already fits every measured path. Report missing evidence and nominal inversions here.
        for index, edge in enumerate(self.graph["edges"]):
            self.timing(edge["kind"], index, edge["source"], edge["target"])
        for index, work in enumerate(self.graph["work"]):
            self.timing("work/" + work["kind"], index, work["start"], work["finish"])
        self.counts.update(self.findings.counts)
        unknown = self.counts["unknown_time"]
        external = self.counts["external_transport_relations"]
        status = "incomplete" if unknown else "external_cost_required" if external else "compatible"
        timing = self.clock.report() | {"status": status, "unknown_constraints": unknown + external}
        return {
            "mapping": "consistent", "timing": timing,
            "events": len(self.graph["events"]), "work": len(self.graph["work"]),
            "coverage": self.coverage,
            "counts": self.counts, "maximum_inversion_ns": self.maximum_inversion_ns,
            "findings": self.findings.examples({"unknown_time"}),
        }


class ProductionEvidence:
    def __init__(self, production, capture, output, details):
        self.production, self.capture, self.output, self.details = production, capture, output, details

    def published_templates(self, release):
        recording = PublishedCapture.from_source(self.capture)
        catalog = release.templates
        observation = catalog.observations.get(recording.attempt_id)
        result = {"recording": recording.attempt_id, "checked_workers": 0}
        if observation is None or observation["deployment"] != recording.inputs["deployment"]:
            return result | {"status": "recording_mismatch"}

        production = self.production.resolve(strict=True)
        paths = read_json(production / "result.json")["templates"]
        if len(paths) != len(observation["shards"]):
            return result | {"status": "worker_count_mismatch"}
        for name, shard in zip(paths, observation["shards"], strict=True):
            native = read_json(asset_path(production, name))
            body = catalog.read_shard(shard)

            # A portable definition keeps its original ID even if its position in the file changes.
            definitions = body.pop("definitions")
            indexed = {definition.pop("id"): definition for definition in definitions}
            expected = dict(enumerate(native.pop("definitions")))
            if len(indexed) != len(definitions) or indexed != expected or (body | shard["header"]) != native:
                return result | {"status": "template_mismatch", "body": shard["body"]}
            result["checked_workers"] += 1
        return result | {"status": "matched"}

    def observe_rank(self, native_case, rank, facts, retain_measurements):

        # SDK/facts belong to this rank; verification can retain the independently checked service evidence.
        graph, native = TraceReader.worker(self.capture, native_case.graphs, rank)
        SourceEvidence.require(native["environment"] == facts["environment"], "clock source environment differs")
        native_evidence, effects = device_effects(graph,
            self.capture, rank, native, facts, self.production / f"work/{rank}-sdk/trace.json",
        )
        host_fences = native_evidence.inspect_host_fences(graph)
        effects["host_fences"] = len(host_fences)
        contents = native_case.observe(rank, graph, native, logical_coordinates=native_evidence.logical_coordinates)
        protocol_posts, external = native_evidence.supplied_protocols(graph)
        measurements = native_evidence.inspect_measurements(external)
        if native_case.joint_execution and measurements["external_transport_endpoints"]:
            raise ValueError("a physical reference cannot substitute single-device staging for transport measurements")
        measurements["checked_device_measurements"] = effects["device_measurements"]
        device_intervals = native_evidence.device_intervals()
        boundaries = native_evidence.inspect_boundaries(graph, device_intervals)
        services, private_streams = native_evidence.inspect_services()
        launches = {} if retain_measurements else None
        submissions, links = native_evidence.inspect_submissions(protocol_posts, launches)
        payload_times = defaultdict(list) if retain_measurements else None
        payload = native_evidence.inspect_payload(
            graph, private_streams, device_intervals, links, launches, payload_times
        )
        generations = native_evidence.event_generations
        signals = native_evidence.inspect_signals(contents, native_case.dependencies[rank], generations)
        stream_roles = native_evidence.inspect_stream_roles(*stream_declarations(native), generations[1])
        report = SourceEvidence(
            facts, graph, native_case.clocks[rank], native_case.logical[rank], self.details,
        ).inspect()
        report.update(native_effects=effects, native_stream_roles=stream_roles, native_payload=payload,
                      native_measurements=measurements, native_boundaries=boundaries, native_services=services,
                      native_submissions=submissions, native_signals=signals)
        if retain_measurements:
            retain_measurements(native_case, rank, facts, graph, native_evidence, host_fences, launches,
                payload_times, report, self.details)
        return report

    def inspect(self, *, retain_measurements=None):
        publication = read_json(self.production / "result.json")
        capture = read_json(self.capture / "capture.json")
        native_case = NativeCase(self.capture, capture, self.production / "work")
        ranks = native_case.workers.observed
        reports = []
        document = {
            "production": str(self.production.resolve()),
            "capture": str(self.capture.resolve()),
            "ranks": reports,
        }
        origin = read_json(self.production / "production.json")["inputs"]["capture_path"]
        if Path(origin).resolve() != self.capture.resolve() or publication["observed_roles"] != ranks:
            return native_case, {**document, "native_case": {"status": "unverified",
                                 "error": "graph production and native capture differ in origin or observed workers"}}

        for rank in ranks:
            try:
                facts = read_json(native_case.graphs / f"{rank}-facts.json")
                report = self.observe_rank(
                    native_case, rank, facts, retain_measurements if native_case.joint_execution else None
                )
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                # A malformed or absent source is a finding; continue the remaining independent ranks.
                report = {"mapping": "unverified", "timing": {"status": "unverified"}, "error": str(error)}
            reports.append({"rank": rank, **report})

        case_report = {"status": "unverified", "error": "one or more worker mappings are unverified"}
        if not native_case.joint_execution:
            case_report = {
                "status": "local_worker_observations",
                "windows": {
                    owner: [list(window.observed) for window in windows]
                    for owner, windows in native_case.workers.windows.items()
                },
            }
        elif all(report["mapping"] == "consistent" for report in reports):
            try:
                native_case.finish()
                case_report = native_case.report(self.details)
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                case_report = {"status": "unverified", "error": str(error)}
        return native_case, {**document, "native_case": case_report}

    @staticmethod
    def qualification(document):
        status = document["native_case"]["status"]
        timing = {"compatible"}
        if status == "local_worker_observations":
            timing.add("external_cost_required")
        workers = bool(document["ranks"]) and all(row["mapping"] == "consistent" and row["timing"]["status"] in timing
                                                  for row in document["ranks"])
        published = "published_templates" not in document or document["published_templates"]["status"] == "matched"
        return {
            "local_templates": workers
            and published
            and status in {"consistent", "incomplete", "local_worker_observations"},
            "joint_reference": workers and status == "consistent",
        }

    def publish(self, document, *, require_joint=False):
        case, workers = document["native_case"], document["ranks"]
        qualification = document["qualification"] = self.qualification(document)
        self.output.mkdir(parents=True)

        report = ReportBuilder("Source evidence")
        report.note(f"Native group `{case['status']}`. Local templates `{qualification['local_templates']}`. "
                    f"Joint reference `{qualification['joint_reference']}`.")
        report.table(("Rank", "Source mapping", "Timing", "Nominal inversions", "Error"),
                     ((row["rank"], row["mapping"], row["timing"]["status"],
                       row.get("counts", {}).get("time_inversion", 0), row.get("error", "")) for row in workers))
        if "error" in case:
            report.note(case["error"])
        if case["status"] == "local_worker_observations":
            report.note("Each worker was observed in its own window. Joint timing requires concurrent workers.")
        if case.get("counts"):
            report.table(("Native finding", "Count"), case["counts"].items())
        report.publish(self.output, "source-evidence", document)
        return 0 if qualification["joint_reference" if require_joint else "local_templates"] else 1
