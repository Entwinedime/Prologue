"""Check retained source evidence independently of template production and prediction."""

import argparse
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path

from prologue.acquisition.recording import PublishedCapture
from prologue.integrations.frameworks.sglang.shared.pipeline import PipelineABI
from prologue.integrations.platforms.ascend.reference import NativeEffectEvidence
from prologue.libraries.release import LibraryRelease, asset_path

from .actions import ActionScopes
from .native_case import NativeCase


class SourceEvidence:
    def __init__(self, facts, graph, clock, logical, details=False):
        self.facts, self.graph = facts, graph
        self.evidence = graph["evidence"]
        self.details = details
        self.counts = Counter()
        self.findings = []
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

        edges = {(edge["source"], edge["target"], edge["kind"]) for edge in graph["edges"]}
        native_edges = set()
        for index, edge in enumerate(facts["dependencies"]):
            source, target = (coordinates[self.coordinate(edge[side])] for side in ("source", "target"))
            self.require(
                source == target or (source, target, edge["kind"]) in edges, f"native dependency {index} is missing"
            )
            if source != target:
                native_edges.add((source, target, edge["kind"]))
        orders = self.validate_coverage(coordinates)
        self.require(edges == native_edges | orders, "projected dependencies omit or add source/serial relations")
        self.require(len(edges) == len(graph["edges"]), "projected dependencies repeat")

        # Projection may commute binary terms, but must preserve every declared formula and condition.
        transformed = {"arguments", "guards", "ports", "kv_accesses", "slot_accesses"}
        for index, (native, projected) in enumerate(zip(facts["scopes"], graph["scopes"], strict=True)):
            self.require({key: value for key, value in native.items() if key not in transformed} ==
                         {key: value for key, value in projected.items() if key not in transformed},
                         f"scope {index} changes source identity, context or communication")
            native_expressions = ActionScopes.declaration_expressions(native)
            projected_expressions = ActionScopes.declaration_expressions(projected)
            for field in native_expressions:
                self.require(
                    native_expressions[field] == projected_expressions[field], f"scope {index} changes source {field}"
                )
            self.require(len(native["ports"]) == len(projected["ports"]), f"scope {index} port count differs")
            for port, target in zip(native["ports"], projected["ports"], strict=True):
                expected = {key: value for key, value in port.items() if key not in {"at", "wait", "rendezvous"}}
                expected["event"] = coordinates[self.coordinate(port["at"])]
                if "wait" in port:
                    expected["wait"] = {
                        "resource": port["wait"]["resource"],
                        "begin": coordinates[self.coordinate(port["wait"]["begin"])],
                    }
                if "rendezvous" in port:
                    expected["rendezvous"] = coordinates[self.coordinate(port["rendezvous"])]

                if "tensor" in expected:
                    self.require(ActionScopes.expression_list_key(expected["tensor"].get("shape_rule"))
                                 == ActionScopes.expression_list_key(target["tensor"].get("shape_rule")),
                                 f"scope {index} port {port['name']} changes its declared shape rule")
                observed = dict(target)
                for value in (expected, observed):
                    if "tensor" in value:
                        value["tensor"] = {key: item for key, item in value["tensor"].items() if key != "shape_rule"}
                self.require(expected == observed, f"scope {index} port {port['name']} differs")
            for field in ("kv_accesses", "slot_accesses"):
                self.require(len(native[field]) == len(projected[field]), f"scope {index} {field} count differs")
                for access, target in zip(native[field], projected[field], strict=True):
                    self.require(
                        ActionScopes.expression_key(access.get("rule"))
                        == ActionScopes.expression_key(target.get("rule")),
                        f"scope {index} {field} changes its declared token rule",
                    )
                    expected = {key: value for key, value in access.items() if key != "rule"}
                    expected.update({side: coordinates[self.coordinate(access[side])] for side in ("start", "finish")})
                    observed = {key: value for key, value in target.items() if key != "rule"}
                    self.require(expected == observed, f"scope {index} {field} changes source access")

    def validate_coverage(self, coordinates):
        ancestors = []
        for index, scope in enumerate(self.facts["scopes"]):
            parent = scope["parent"]
            self.require((parent is None and index == 0) or (parent is not None and 0 <= parent < index),
                         f"scope {index} has an invalid source parent")
            ancestors.append({index} | (ancestors[parent] if parent is not None else set()))

        spans, boundaries, touching = defaultdict(list), defaultdict(list), defaultdict(set)
        required = set()
        for index, span in enumerate(self.facts["spans"]):
            resource = self.facts["resources"][span["resource"]]
            owner = None if resource["serial"] else index
            spans[span["resource"], owner].append((index, span))
            for position in (span["begin"], span["end"]):
                required.add((span["resource"], position, owner))
                touching[span["resource"], position, owner].add(span["scope"])
        for scope in self.facts["scopes"]:
            for port in scope["ports"]:
                required.add(self.coordinate(port["at"]))
                if "wait" in port:
                    required.add(self.coordinate(port["wait"]["begin"]))
                if "rendezvous" in port:
                    required.add(self.coordinate(port["rendezvous"]))
            for access in scope["kv_accesses"] + scope["slot_accesses"]:
                required.update(self.coordinate(access[side]) for side in ("start", "finish"))
        for edge in self.facts["dependencies"]:
            required.update(self.coordinate(edge[side]) for side in ("source", "target"))
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

        self.counts[issue] += 1
        if self.details or self.counts[issue] <= 3:
            self.findings.append({
                "issue": issue, "kind": kind, "index": index, "source": source, "target": target,
                "source_ns": begin, "target_ns": end, "difference_ns": difference,
            })

    def inspect(self):
        self.validate_mapping()

        # The case clock already fits every measured path. Report missing evidence and nominal inversions here.
        for index, edge in enumerate(self.graph["edges"]):
            self.timing(edge["kind"], index, edge["source"], edge["target"])
        for index, work in enumerate(self.graph["work"]):
            self.timing("work/" + work["kind"], index, work["start"], work["finish"])
        unknown = self.counts["unknown_time"]
        external = self.counts["external_transport_relations"]
        status = "incomplete" if unknown else "external_cost_required" if external else "compatible"
        timing = self.clock.report() | {"status": status, "unknown_constraints": unknown + external}
        return {
            "mapping": "consistent", "timing": timing,
            "events": len(self.graph["events"]), "work": len(self.graph["work"]),
            "coverage": self.coverage,
            "counts": dict(self.counts), "maximum_inversion_ns": self.maximum_inversion_ns, "findings": self.findings,
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
        paths = json.loads((production / "result.json").read_bytes())["templates"]
        if len(paths) != len(observation["shards"]):
            return result | {"status": "worker_count_mismatch"}
        for name, shard in zip(paths, observation["shards"], strict=True):
            native = json.loads(asset_path(production, name).read_bytes())
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
        graph = json.loads((self.production / f"work/{rank}-graph.json").read_text())
        native = json.loads((self.capture / f"rank-{rank}.json").read_text())
        SourceEvidence.require(native["environment"] == facts["environment"], "clock source environment differs")
        native_evidence = NativeEffectEvidence(
            self.capture, rank, native, facts, self.production / f"work/{rank}-sdk/trace.json",
        )
        effects = native_evidence.inspect(graph)
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
        stream_roles = native_evidence.inspect_stream_roles(*PipelineABI.stream_declarations(native), generations[1])
        report = SourceEvidence(
            facts, graph, native_case.clocks[rank], native_case.logical[rank], self.details,
        ).inspect()
        report["native_effects"] = effects
        report["native_stream_roles"] = stream_roles
        report["native_payload"] = payload
        report["native_measurements"] = measurements
        report["native_boundaries"] = boundaries
        report["native_services"] = services
        report["native_submissions"] = submissions
        report["native_signals"] = signals
        if retain_measurements:
            from .fixed_context import FixedHostContext

            records = {key: facts[key] for key in ("resources", "environment", "measurements")}
            # Timed work and published physical signals retain their actual measured endpoints.
            timed = {piece[side] for piece in graph["work"] for side in ("start", "finish")}
            timed.update(port["event"] for scope in graph["scopes"] for port in scope["ports"]
                         if port["use"] == "signal" and port["signal"].startswith("event/"))
            records["observer_joins"] = native_evidence.observer_joins - timed
            releases = native_evidence.sdk.release_evidence(
                graph, launches, native_evidence.queues, native_case.clocks[rank], records["observer_joins"]
            )
            omitted = {record for release in releases.values() for record in release.omitted}
            owner = native_case.ranks[rank]
            deliveries = zip(native_case.manifest["instances"], native_case.submissions, strict=True)
            delivery = next(submitted for instance, submitted in deliveries if instance["id"] == owner)
            context = FixedHostContext(self.capture, rank, native_evidence, delivery)
            calls = native_case.cpu.native_calls[rank]
            measured = context.project(
                native_evidence, graph, host_fences, omitted, self.details, calls, native_case.cpu.native_times
            )
            delays, waits, idle_inputs, posts, queues, protocol = measured
            records["fixed_cpu_calls"] = protocol
            report["native_host_queues"] = queues
            records["fixed_host_delays"] = delays
            records["fixed_idle_inputs"] = idle_inputs
            records["host_fences"] = host_fences
            records["fixed_driver_posts"] = posts
            records["device_releases"] = releases
            records["device_payload_times"] = payload_times
            records["fixed_host_waits"] = waits
            records["fixed_host_context"] = context
            records["kernel_threads"] = context.active.kernel_threads
            records["unbound_queue_notifications"] = (
                queues["queue_completion_notifications"] - queues["bound_to_fixed_work"]
            )

            native_case.measurements[rank] = records
        return report

    def inspect(self, *, retain_measurements=False):
        publication = json.loads((self.production / "result.json").read_text())
        capture = json.loads((self.capture / "capture.json").read_text())
        native_case = NativeCase(self.capture, capture, self.production / "work")
        ranks = native_case.workers.observed
        reports = []
        document = {
            "production": str(self.production.resolve()),
            "capture": str(self.capture.resolve()),
            "ranks": reports,
        }
        origin = json.loads((self.production / "production.json").read_text())["inputs"]["capture_path"]
        if Path(origin).resolve() != self.capture.resolve() or publication["observed_roles"] != ranks:
            return native_case, {**document, "native_case": {"status": "unverified",
                                 "error": "graph production and native capture differ in origin or observed workers"}}

        for rank in ranks:
            try:
                facts = json.loads((native_case.graphs / f"{rank}-facts.json").read_text())
                report = self.observe_rank(
                    native_case, rank, facts, retain_measurements and native_case.joint_execution
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
        reports, case_report = document["ranks"], document["native_case"]
        eligibility = document["qualification"] = self.qualification(document)
        self.output.mkdir(parents=True)
        (self.output / "source-evidence.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        lines = [
            "# Source Evidence Check",
            "",
            "This check compares raw declarations, interpreted records, and the coarse graph. It covers the "
            "execution environment, call hierarchy and context, communication, retained rules, work and ordering "
            "intervals, boundary ownership, geometry, dependencies, state access, and agreement between measured "
            "time and graph causality. Each work interval must appear exactly once. Waits and payload-free "
            "intervals contribute ordering only. Instrumentation completeness, applicability to target "
            "configurations, template generalization, and target predictions require separate evidence.",
            "",
            "Native effects are checked against SDK calls, device tasks, runtime records, and queue lifecycles "
            "from the same run. SDK records or a shared native submission group identify device work. A copy "
            "without an SDK task requires evidence of a synchronous driver transfer on the original buffer and "
            "contributes only a caller interval. Explicit synchronous copies also require a measured caller "
            "interval and a dependency on the actual return fence.",
            "",
            "| Rank | Source mapping | Async copies / sync copies / stream waits | Clock calibration | Nominal "
            "time inversions | Largest nominal inversion (ns) |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
        labels = {
            "consistent": "Consistent", "compatible": "Compatible within calibration bounds",
            "external_cost_required": "Local clocks compatible with external transport timing pending",
            "inconsistent": "No shared calibration", "incomplete": "Missing measurements", "unverified": "Unverified",
        }
        for report in reports:
            effects = report.get("native_effects")
            native_counts = (
                f"{effects.get('copies', 0)} / {effects.get('synchronous_copies', 0)} / "
                f"{effects.get('stream_waits', 0)}"
                if effects is not None
                else "-"
            )
            lines.append(
                f"| {report['rank']} | {labels[report['mapping']]} | {native_counts} | "
                f"{labels[report['timing']['status']]} | {report.get('counts', {}).get('time_inversion', '-')} | "
                f"{report.get('maximum_inversion_ns', '-')} |"
            )
        for report in reports:
            if "error" in report:
                lines.extend(["", f"Rank {report['rank']} could not be checked. {report['error']}", ""])
            elif "native_boundaries" in report:
                boundaries, signals = report["native_boundaries"], report["native_signals"]
                measurements = report["native_measurements"]
                services = report["native_services"]
                streams, payload = report["native_stream_roles"], report["native_payload"]
                lines.extend(
                    [
                        "",
                        f"### Rank {report['rank']}",
                        "",
                        f"Checked {measurements['checked_host_measurements']} host timestamps and "
                        f"{measurements['checked_device_measurements']} device endpoints against raw records. "
                        f"{measurements['unmeasured_device_endpoints']} stream boundaries lack measurements, and "
                        f"{measurements['logical_wait_boundaries']} logical wait boundaries have no device "
                        f"submission. {measurements['external_transport_endpoints']} transport boundaries require "
                        "external timing. Synchronous transfers retain the driver's host timestamps, and device "
                        "return fences retain SDK timestamps.",
                        "",
                        f"Checked {streams['named_streams']} named streams, "
                        f"{payload.get('retained_payload_tasks', 0)} retained SDK payload tasks, and "
                        f"{payload.get('opaque_service_tasks', 0)} tasks in private service queues. Checked "
                        f"{boundaries.get('host_intervals', 0)} raw host intervals, "
                        f"{boundaries.get('service_intervals', 0)} service intervals, and "
                        f"{boundaries.get('declared_state_accesses', 0)} declared state accesses.",
                        "",
                        f"Observed {services['synchronous_services']} synchronous services, "
                        f"{services['private_queues']} private queues, and {services['submitted_tasks']} native "
                        f"submissions. The SDK evidence includes {signals.get('exported_completions', 0)} exported "
                        f"completions, {signals.get('signal_waits', 0)} waits, and "
                        f"{signals.get('true_completion_reads', 0)} completion queries that returned true. The "
                        f"{boundaries.get('derived_state_accesses', 0)} address-derived PD state accesses require "
                        "pool and transfer evidence.",
                    ]
                )
        lines.extend(["", f"Native group result `{case_report['status']}`."])
        if "error" in case_report:
            lines.extend(["", case_report["error"]])
        elif case_report["status"] == "local_worker_observations":
            lines.extend(["", "Each worker's SDK mapping is checked in its recorded window. "
                          "Target peer timing requires profiles from a shared execution on the physical topology."])
        else:
            model = case_report["model_communication"]
            payload = case_report["pd_payload"]
            lines.extend(
                [
                    "",
                    f"Checked {model['communication_exchanges']} model communication rounds, "
                    f"{case_report['model_links']} cross-rank relations, and {case_report['pd_links']} required PD "
                    f"relations. CPU protocol completeness is `{case_report['cpu_protocol']['complete']}`. "
                    f"Observed {case_report['physical_bindings']['observed_request_bindings']} physical request "
                    "bindings.",
                    "",
                    f"PD payload checks cover {payload.get('scopes_checked', 0)} transfer scopes, "
                    f"{payload.get('kv_accesses_checked', 0)} KV accesses, and "
                    f"{payload.get('metadata_accesses_checked', 0)} metadata accesses. Registered buffers, copied "
                    "bytes, and request page tables establish the model layer and K/V component at each end. "
                    "Tensor values and repeated copy counts require separate evidence.",
                ]
            )
            for finding in case_report["findings"]:
                lines.extend(["", f"`{finding['category']}` {finding.get('reason', finding['kind'])}"])
        if "published_templates" in document:
            published = document["published_templates"]
            lines.extend(
                [
                    "",
                    f"Released templates: {published['status']}. Matched {published['checked_workers']} original "
                    "worker bodies.",
                ]
            )
        lines.extend(
            [
                "",
                "Nominal inversions use original midpoint timestamps. Each rank uses one calibration across all "
                "measured relations, including paths through logical waits. The original API, its observed "
                "prerequisites, and the absence of physical submissions qualify logical waits and any data handed "
                "off at their returns. Local work, resource accesses, and recorded signals require measured "
                "endpoints. A separately observed worker can establish a transport operand and its consumer while "
                "its target transport cost remains external. Compatibility establishes a feasible calibration on "
                "the original graph and timestamps. Exact offsets, capture completeness, and SDK attribution "
                "require their own evidence. `unknown_time` marks a required measurement gap.",
                "",
                "Exit status 0 requires consistent worker mappings and local timing evidence. A joint target "
                "reference additionally requires consistent physical group evidence and calibration. The group "
                "check uses native records for participant, CPU protocol, and binding evidence. The default JSON "
                "lists up to three nominal timing findings per category. `--details` includes every location.",
                "",
            ]
        )
        (self.output / "source-evidence.md").write_text("\n".join(lines))
        return 0 if eligibility["joint_reference" if require_joint else "local_templates"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production", type=Path, required=True, help="published produce output with retained intermediates"
    )
    parser.add_argument(
        "--capture", type=Path, required=True, help="original native source containing independent device clock samples"
    )
    parser.add_argument(
        "--libraries", type=Path, help="also compare the released templates with this original production"
    )
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--details", action="store_true", help="include all timing findings")
    arguments = parser.parse_args()
    check = ProductionEvidence(arguments.production, arguments.capture, arguments.output, arguments.details)
    _, document = check.inspect()
    if arguments.libraries is not None:
        document["published_templates"] = check.published_templates(LibraryRelease(arguments.libraries))
    raise SystemExit(check.publish(document))


if __name__ == "__main__":
    main()
