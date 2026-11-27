"""Build every native worker from its recording and qualify the projected execution graph."""

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import content_id, read_json
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.libraries.release import asset_path
from prologue.host.libraries.templates import TemplatePublication
from prologue.verification.core import RawCaseSource
from prologue.verification.core.actions import ActionScopes
from prologue.verification.core.adapters import device_effects, stream_declarations
from prologue.verification.core.native_case import NativeCase
from prologue.verification.engine import FindingLedger, ReportBuilder, TraceReader


class SourceEvidence:
    """Reconstruct the native partition and compare it with the projected graph at every qualified coordinate."""

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
            self.partition_lane(key, points, spans[key], touching, ancestors, expected, orders)

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

    def partition_lane(self, key, points, spans, touching, ancestors, expected, orders):
        """Rebuild the native partition and check every shared boundary's semantic owner."""

        resource = self.facts["resources"][key[0]]
        payload = resource["payload"]
        payload_ends = [end for _, end in payload]
        pending = iter(sorted(spans, key=lambda item: item[1]["begin"]))
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


def template_matches(catalog, shard, observed):
    body = catalog.read_shard(shard)

    # A portable definition keeps its source ID even if its position in the file changes.
    definitions = body.pop("definitions")
    indexed = {definition.pop("id"): definition for definition in definitions}
    header_body = {key: value for key, value in observed.items() if key != "definitions"}
    return (len(indexed) == len(definitions) and indexed == dict(enumerate(observed["definitions"]))
            and (body | shard["header"]) == header_body)


class LocalTemplateEvidence:
    """Check the released fragments against the qualified products of their native cut."""

    def __init__(self, recording, release, model, resources):
        self.recording, self.release, self.model, self.resources = recording, release, model, resources

    def inspect(self, scratch):
        recording = self.recording
        catalog = self.release.templates
        observation = catalog.observations.get(recording.attempt_id)
        if observation is None or observation["deployment"] != recording.inputs["deployment"]:
            raise ValueError("released fragments do not belong to this native cut")

        shards = observation["shards"]
        if not shards:
            raise ValueError("native cut has no released fragments")
        needed = {shard["product"].partition("-")[0] for shard in shards}
        products = {}
        for name, digest in recording.receipt["assets"].items():
            identity = content_id([recording.attempt_id, name])
            if identity not in needed:
                continue
            product = read_json(asset_path(recording.session, name))
            if (content_id(product) != digest or recording.attempt_id not in product["native_recordings"]
                    or product["acquisition_rule"] != recording.receipt["rule"]):
                raise ValueError(f"qualified local product changed: {name}")

            if "participants" in product:
                for worker in product["participants"]:
                    rank = worker["environment"]["parallel"]["rank"]
                    products[f"{identity}-{rank}"] = worker
            else:
                products[identity] = product

        for shard in shards:
            identity = shard["product"]
            if identity not in products:
                raise ValueError(f"template has no qualified local product: {identity}")
            facts = products[identity]
            template = TemplatePublication.compile(self.model, scratch, identity, facts, self.resources)
            sampling = read_json(asset_path(catalog.directory, shard["sampling"])) if "sampling" in shard else None
            initial = shard.get("initialization", {})
            if (not template_matches(catalog, shard, template) or sampling != facts.get("sampling")
                    or any(facts[key] != value for key, value in initial.items())):
                raise ValueError(f"released template differs from its qualified local product: {identity}")

        return len(shards)

    def run(self, output):
        output.mkdir(parents=True)
        result = {"recording": self.recording.attempt_id, "scope": "local_acquisition_cut"}
        try:
            with TemporaryDirectory(dir=output) as temporary:
                result.update(status="matched", checked_fragments=self.inspect(Path(temporary)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="unverified", error=str(error))

        matched = result["status"] == "matched"
        result["qualification"] = {"local_templates": matched, "joint_reference": False}
        report = ReportBuilder("Local template source")
        report.note(result.get("error", f"Checked {result.get('checked_fragments', 0)} released fragments."))
        report.publish(output, "source-evidence", result)
        return 0 if matched else 1


class ProductionEvidence:
    """Qualify each projected rank and then establish whether the original observations support a joint
    reference.
    """

    def __init__(self, production, source: RawCaseSource, output, details):
        self.production, self.source, self.output, self.details = production, source, output, details
        self.capture = source.directory

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
            if not template_matches(catalog, shard, native):
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
        native_case = NativeCase(self.source, self.production / "work")
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
