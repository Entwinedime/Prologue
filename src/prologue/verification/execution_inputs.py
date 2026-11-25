"""Prepare native execution conditions and independently check their actual consumption."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from prologue.runtime.io import read_json, write_json

from .actions import ActionIdentity
from .context_values import ContextValues
from .dependencies import ActionBoundaries
from .ingress import IngressEvidence
from .native_case import NativeCase
from .source_clock import ClockRealization


class NativeInputs:
    def __init__(self, capture, graphs, output, layouts=None, *, peer_handoffs=False):
        self.capture, self.graphs, self.output = capture, graphs, output
        self.layouts, self.peer_handoffs = layouts, peer_handoffs
        self.steps = defaultdict(list)
        self.cpu_positions = {}
        self.cpu_consumers = set()
        self.origins, self.ingress = {}, {}
        self.counts = Counter()
        self.contexts = ContextValues()
        self.layout_values = ContextValues()
        self.realization = None

    @staticmethod
    def window(begin, end, origin):
        if begin > end or end < origin:
            raise ValueError("native condition has a reversed or pre-workload interval")
        return {"begin_ns": max(begin - origin, 0), "end_ns": end - origin}

    @staticmethod
    def port(name, direction, window):
        return {"at": {"port": name, "direction": direction}, "window": window}

    def actions(self, case, rank, realization, origin):
        group, clock = case.groups[rank], case.clocks[rank]
        source, raw, identities = case.sources[rank]
        graph, evidence = source.graph, source.graph["evidence"]
        if evidence["clock"] != "monotonic_raw_ns":
            raise ValueError("native input conditions require the common RAW clock")
        layouts = None if self.layouts is None else read_json(self.layouts / f"{rank}-layouts.json")
        if layouts is not None and set(layouts) != {str(index) for index in identities}:
            raise ValueError("timing layouts and independently observed action scopes differ")
        if layouts is not None and sum(len(layout["identity"][-1]) for layout in layouts.values()) != len(graph["work"]):
            raise ValueError(f"rank {rank} action cuts omit or repeat native work")
        services = case.measured_service_pairs(rank, graph, raw) if layouts is not None else set()
        self.counts["unmeasured_work_finishes"] += sum(evidence["events"][work["finish"]]["time_ns"] is None for work in graph["work"])
        for index, identity in identities.items():
            scope = graph["scopes"][index]
            entered = next(port["event"] for port in scope["ports"] if port["name"] == "call.enter")
            begin = evidence["events"][entered]["time_ns"]
            if begin is None:
                raise ValueError(f"rank {rank} action {index} event {entered} has no measured entry")
            if "work" not in identity:
                raise ValueError(f"rank {rank} action {index} has no complete logical identity")
            record = {"operation": "action", "identity": self.contexts.encode({
                name: identity[name] for name in ("action", "parameters", "work")
            }), "source": {"scope": index}}
            if layouts is not None:
                layout = layouts[str(index)]
                record["layout"] = self.layout_values.intern(layout["identity"])

                # The full work signature selects the graph. Timing consumes only its two event coordinates.
                work_endpoints = (interval[2:4] for interval in layout["identity"][-1])
                record["services"] = [[start, finish] for start, finish in work_endpoints
                                      if (layout["events"][start], layout["events"][finish]) in services]
                record["endpoints"] = []
                for event in layout["events"]:
                    measured = clock.window(event)
                    if measured is None:
                        if event not in case.logical[rank]:
                            raise ValueError(f"rank {rank} action {index} event {event} lacks measurement or a verified order witness")
                        time_ns = None
                        self.counts["logical_boundaries"] += 1
                    else:
                        selected = realization.timestamp(rank, event)
                        time_ns = self.window(selected, selected, origin)["begin_ns"]
                        self.counts["measured_boundaries"] += 1
                        self.counts["calibrated_boundary_ranges"] += measured[0] != measured[1]
                    record["endpoints"].append([event, time_ns])
                self.counts["measured_work"] += len(layout["identity"][-1])
                self.counts["isolated_observed_services"] += len(record["services"])

            if layouts is None or self.peer_handoffs:
                boundaries = []
                for boundary in case.completion_boundaries(rank, index, peer_handoffs=self.peer_handoffs):
                    event = boundary["event"]
                    measured = clock.window(event)
                    if measured is None:
                        self.counts["unmeasured_boundaries"] += 1
                        continue
                    selected = realization.timestamp(rank, event)
                    condition = self.window(selected, selected, origin)
                    if layouts is None:
                        self.counts["measured_boundaries"] += 1
                        self.counts["calibrated_boundary_ranges"] += measured[0] != measured[1]
                    else:
                        self.counts["common_port_conditions"] += 1
                    boundaries.append({"at": boundary["at"], "window": condition})
                bindings = source.invocation_bindings(index, group)
                if bindings:
                    record["bindings"] = bindings
                    self.counts["bound_invocations"] += 1
                record.update(service_ns=None, boundaries=boundaries)
                if layouts is not None:
                    record["host_tails"] = []
                    for tail in layout["host_tails"]:
                        if clock.window(tail["event"]) is None:
                            continue
                        selected = realization.timestamp(rank, tail["event"])
                        record["host_tails"].append({**tail, "window": self.window(selected, selected, origin)})
            self.steps[group].append((begin, record, None))

    def input_reads(self, instance, rank, raw, submitted):
        if not raw["request_ingress"]:
            return
        name, group = instance["id"], f"{instance['id']}/{rank}"
        if name in self.ingress:
            raise ValueError("native input has more than one external ingress per instance")
        self.ingress[name] = submitted.receives(raw["request_ingress"])
        if IngressEvidence.attempts(raw)["status"] != "observed":
            raise ValueError("native input reads lack their complete entering contexts")
        origin = submitted.origin_ns
        reads = IngressEvidence.reads(name, raw, submitted.aliases, origin)
        for number, (interval, read) in enumerate(zip(raw["request_input_attempts"], reads, strict=True)):
            self.steps[group].append((interval["begin_raw_ns"], {
                "operation": "input", "identity": self.contexts.encode(read["identity"]), "source": {"input": number},
                "window": {"begin_ns": read["time_ns"], "end_ns": read["time_ns"]},
            }, None))

    def cpu(self, evidence, rank, origin):
        group = evidence.groups[rank]
        for call in evidence.cpu.native_calls[rank]:
            if not call.execution["in_flow"]:
                continue
            begin, end = (evidence.cpu.native_times[point] for point in (call.entered, call.returned))
            position = (call.key, rank) if call.phase == "post" else None
            boundaries = [
                self.port("call.enter", "input", self.window(begin, begin, origin)),
                self.port("call.return", "output", self.window(end, end, origin)),
            ]
            if position is not None:
                if call.posted is None:
                    raise ValueError(f"serving CPU post {call.key} at rank {rank} lacks an actual transport boundary")
                posted = evidence.cpu.native_times[call.posted]
                boundaries.append(self.port("transport.post", "output", self.window(posted, posted, origin)))
            if call.phase == "wait" or call.execution["operation"] in {"send", "recv"}:
                self.cpu_consumers.add((call.key, rank))
            self.steps[group].append((begin, {
                "operation": "cpu", "identity": self.contexts.encode(call.identity()), "source": call.source(),
                "service_ns": None, "boundaries": boundaries,
            }, position))

    def transfers(self, evidence):
        result = []
        for key, peers in evidence.cpu.messages["native"].items():
            positions = [self.cpu_positions[key, rank] for rank in peers if (key, rank) in self.cpu_positions]
            consumers = [rank for rank in peers if (key, rank) in self.cpu_consumers]
            if not positions and not consumers:
                continue
            origins = {self.origins[evidence.groups[rank]] for rank in peers}
            if len(origins) != 1:
                raise ValueError("native CPU message crosses unrelated workload clocks")
            origin, = origins
            timing = evidence.cpu.native_readiness(peers)
            if timing is None or None in timing[1].values():
                raise ValueError(f"CPU transfer {key} lacks independent readiness on both endpoints")
            _, windows = timing
            if not positions:
                if any(windows[rank][1] > origin for rank in consumers):
                    raise ValueError(f"CPU transfer {key} becomes ready during serving without a timed post")
                continue

            # The pairing uses already checked posting conditions, before a consumer can advance.
            posts = sorted(positions, key=lambda row: (row["group"], row["step"]))
            result.append({"posts": posts, "ready": {
                side: self.window(*windows[rank], origin) for side, rank in zip(("send", "receive"), key[1:3], strict=True)
            }})
        return result

    def run(self, reference=None):
        self.output.mkdir(parents=True)
        report = {"capture": str(self.capture.resolve()), "graphs": str(self.graphs.resolve()), "status": "unverified"}
        try:
            evidence = NativeCase.read(self.capture, self.graphs) if reference is None else reference
            if self.peer_handoffs and not any(source[0] != target[0] for source, target, _ in evidence.links):
                raise ValueError("peer handoff timing requires a verified cross-worker dependency")
            for instance, session, submitted in zip(evidence.manifest["instances"], evidence.sessions, evidence.submissions, strict=True):
                if set(instance["observed_roles"]) != set(session["native_participants"]):
                    raise ValueError("native input needs every actual participant")
                origin = submitted.origin_ns
                for rank in session["native_participants"]:
                    group = f"{instance['id']}/{rank}"
                    self.origins[group] = origin
                    _, raw, _ = evidence.sources[rank]
                    self.input_reads(instance, rank, raw, submitted)
                    self.cpu(evidence, rank, origin)
                if instance["id"] not in self.ingress:
                    raise ValueError("native case omits external ingress")
            if len(set(self.origins.values())) != 1:
                raise ValueError("native serving instances have different workload origins")
            native_report = evidence.report(False)
            report["native_evidence"] = native_report
            if evidence.graph is None or native_report["status"] == "differences":
                raise ValueError("native source has no consistent causal graph")
            self.realization = ClockRealization(evidence.clocks, evidence.graph, evidence.cpu.native_times).solve()
            for rank in evidence.sources:
                group = evidence.groups[rank]
                self.actions(evidence, rank, self.realization, self.origins[group])
            actors = {}
            for group in self.origins:
                ordered = sorted(self.steps[group], key=lambda step: step[0])
                if len({begin for begin, _, _ in ordered}) != len(ordered):
                    raise ValueError(f"native {group} has unordered operation entries at the same RAW timestamp")
                actors[group] = {"steps": [step for _, step, _ in ordered]}
                for position, (_, _, post) in enumerate(ordered):
                    if post is not None:
                        self.cpu_positions[post] = {"group": group, "step": position}
            notifications = [
                {"identity": {"group": row["group"], "name": name, "work": {"request": row["request"]}},
                 "window": self.window(row[field]["begin_raw_ns"], row[field]["end_raw_ns"], self.origins[row["group"]])}
                for row in evidence.readiness["endpoints"]
                for name, field in (("kv_ready", "reservation_update"), ("kv_success", "completion_update"))
                if row[field] is not None and (name != "kv_ready" or row["remote_reservation"])
            ]
            for row in evidence.readiness["reservation_receipts"]:
                group, time = row["recipient"], row["raw_ns"]
                notifications.append({
                    "identity": {"group": group, "name": "kv_reservation_received",
                                 "work": {"request": row["request"], "publisher": row["publisher"]}},
                    "window": self.window(time, time, self.origins[group]),
                })
            document = {"context_values": self.contexts.values, "actors": actors, "ingress": self.ingress,
                        "notifications": notifications, "transfers": self.transfers(evidence)}
            if self.layouts is not None:
                document["layouts"] = self.layout_values.values
            report.update(status="partial" if self.counts["unmeasured_boundaries"] or self.counts["unmeasured_work_finishes"]
                          or native_report["status"] != "consistent" else "extracted",
                          clock_realization=self.realization.report(), counts=dict(self.counts),
                          steps={group: len(actor["steps"]) for group, actor in actors.items()})
            if report["status"] == "extracted":
                write_json(self.output / "execution-inputs.json", document, replace=False)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            report.update(status="unverified", error=str(error))
        report["timing_scope"] = "peer_handoffs" if self.peer_handoffs else "observed_execution"
        report["scope"] = "Measured endpoints use one feasible device-clock realization rounded upward to integer nanoseconds. Independently verified elided waits supply causal witnesses without time constraints. Original calibration uncertainty and CPU wire and notification bounds remain."
        (self.output / "execution-inputs-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report


class ConsumedInputs:
    """Prove consumed prefixes from native observations, independently of the supplied condition file."""

    def __init__(self, case, ingress, realization, *, peer_handoffs=False):
        self.groups, self.ingress = case.groups, ingress
        self.case = case
        self.peer_handoffs = peer_handoffs
        self.service_pairs = {}
        self.ranks = {group: rank for rank, group in case.groups.items()}
        self.origin_ns = ingress.origin_ns
        self.ports, self.cpu_calls = {}, {}
        self.cpu_messages = {}
        self.bindings = {}
        self.boundaries = defaultdict(dict)
        self.realization = realization
        self.expected = None
        self.positions = Counter()
        self.pending, self.reads = {}, {}
        self.input_pairs = defaultdict(dict)
        self.counts = Counter()
        self.measured_graph = None

    def prepare(self):
        ordered = defaultdict(list)
        for rank, (source, raw, identities) in self.case.sources.items():
            group = self.groups[rank]

            # Original CPU entries establish order; graph scope references retain their independently checked raw origins.
            host = {int(row["name"].split("/")[1]): row["begin_raw_ns"]
                    for row in raw["host_observation"]["intervals"] if row["name"].startswith("prologue.scope/")}
            host.update({query["scope"]: query["begin_raw_ns"] for query in raw["state_queries"]})
            for index, identity in identities.items():
                self.ports[group, index] = tuple(self.case.completion_boundaries(rank, index, peer_handoffs=self.peer_handoffs))
                if bindings := source.invocation_bindings(index, group):
                    self.bindings[group, index] = bindings
                origins = source.graph["scopes"][index]["bindings"]["source_scopes"]
                begin = min(raw["scopes"][origin]["host_interval"]["begin_raw_ns"] if "service" in raw["scopes"][origin] else host[origin]
                            for origin in origins)
                ordered[group].append((begin, {"operation": "action", "identity": {key: identity[key] for key in ("action", "parameters", "work")},
                                              "source": {"scope": index}}))

            for call in self.case.cpu.native_calls[rank]:
                if not call.execution["in_flow"]:
                    continue
                begin, end = (self.case.cpu.native_times[point] for point in (call.entered, call.returned))
                self.cpu_calls[group, call.key, call.phase] = call
                ordered[group].append((begin, {
                    "operation": "cpu", "identity": call.identity(), "source": call.source(),
                }))

            if raw["request_ingress"]:
                instance = group.rsplit("/", 1)[0]
                for index, (interval, read) in enumerate(zip(raw["request_input_attempts"], self.ingress.native_reads[instance], strict=True)):
                    ordered[group].append((interval["begin_raw_ns"], {
                        "operation": "input", "identity": read["identity"], "source": {"input": index},
                    }))

        self.expected = {}
        for group, records in ordered.items():
            records.sort(key=lambda item: item[0])
            if any(left[0] >= right[0] for left, right in zip(records, records[1:])):
                raise ValueError("native condition witnesses need strictly ordered measured CPU entries")
            self.expected[group] = [record for _, record in records]

    def observe(self, row):
        kind = row["kind"]
        if self.measured_graph is not None:
            rank, root, target, event_ids, matches, durations = self.measured_graph
            if kind in {"boundary_time", "logical_boundary"} and "source_event" in row:
                target_event, source_event = row["event"], row["source_event"]
                if target_event in matches:
                    raise ValueError("full timing repeats a target event")
                time_ns = None
                if kind == "boundary_time":
                    time_ns = row["begin_ns"]
                    if time_ns != row["end_ns"]:
                        raise ValueError("a matched native timestamp must use a single realized time")
                matches[target_event] = source_event, time_ns
            elif kind != "boundary_time":
                if matches:
                    if self.peer_handoffs and len(matches) < len(event_ids):
                        self.check_host_tails(rank, root, target, event_ids, matches)
                    else:
                        self.check_measured_graph(rank, root, target, event_ids, matches, durations)
                self.measured_graph = None

        if kind == "condition_used":
            if self.expected is None:
                self.prepare()
                if self.realization is None:
                    self.realization = ClockRealization(self.case.clocks, self.case.graph, self.case.cpu.native_times)
            group = row["group"]
            position = self.positions[group]
            if (row.keys() != {"kind", "group", "step", "operation", "source"} or type(row["step"]) is not int
                    or row["step"] != position or position >= len(self.expected[group]) or group in self.pending):
                raise ValueError(f"invalid consumed execution condition position: {group} at {position}")

            # The following graph or input read supplies the actual identity; the receipt only selects its original condition.
            record = self.expected[group][position]
            if row["operation"] != record["operation"] or row["source"] != record["source"]:
                raise ValueError(f"consumed execution condition differs from its native causal prefix: {group} at {position}")
            self.positions[group] += 1
            self.counts[record["operation"]] += 1
            if record["operation"] == "input":
                instance = record["identity"]["instance"]
                if instance in self.reads:
                    raise ValueError("input condition was consumed before the preceding read returned")
                self.reads[instance] = record["source"]["input"], record["identity"]
            else:
                self.pending[group] = record
        elif kind == "input_read" and self.expected is not None:
            instance = row["identity"]["instance"]
            if instance in self.reads:
                source, identity = self.reads.pop(instance)
            elif not row["in_flow"]:
                return
            else:
                raise ValueError("active input read has no consumed condition")
            if identity != row["identity"]:
                raise ValueError("input result belongs to a different consumed condition")
            self.input_pairs[instance][len(self.ingress.actual_reads[instance])] = source

        elif kind == "completion" and self.expected is not None:
            raw_ns = row["time_ns"] + self.origin_ns
            for (rank, event), declared in self.boundaries.pop(row["event"], {}).items():
                if rank is None:
                    if raw_ns != event:
                        raise ValueError("CPU protocol completion changes its original RAW boundary")
                elif event in self.case.logical[rank]:
                    self.counts["logical_boundaries"] += 1
                    continue
                elif declared is None:
                    self.realization.condition(rank, event, raw_ns)
                elif declared != raw_ns:
                    raise ValueError(f"rank {rank} event {event} completes outside its declared native timestamp")
                self.counts["timed_boundaries"] += 1
        elif kind == "native_protocol" and row["record"]["phase"] == "ready":
            record = row["record"]
            predicted = tuple(record["key"])
            if record["event"] is not None and predicted in self.cpu_messages:
                native = self.cpu_messages[predicted]
                _, windows = self.case.cpu.native_readiness(self.case.cpu.messages["native"][native])
                begin, end = windows[self.ranks[row["group"]]]
                if not begin <= row["time_ns"] + self.origin_ns <= end:
                    raise ValueError("CPU readiness differs from the original transport callback")
                self.counts["cpu_readiness_endpoints"] += 1

    def matched_invocation(self, group, root):
        record = self.pending.get(group)
        if record is None:
            raise ValueError(f"no native invocation witness for {group}")
        if root["role"] == "cpu_service":
            execution = root["bindings"]["cpu_protocol"]["execution"]
            operation = "cpu"
            identity = {"operation": execution["operation"], "work": ActionIdentity.work_identity(root["work"])}
        else:
            operation = "action"
            identity = {key: value for key, value in ActionIdentity.record(root, group).items() if key != "group"}
        if record["operation"] != operation or record["identity"] != identity:
            raise ValueError("instantiated graph differs from its consumed execution condition")
        return record

    def graph(self, group, row, contents):
        if self.expected is None:
            return None
        root = row["graph"]["scopes"][0]
        record = self.matched_invocation(group, root)
        del self.pending[group]
        ports = {(port["name"], port["direction"]): row["event_ids"][port["event"]] for port in root["ports"]}
        if record["operation"] == "action":
            scope = record["source"]["scope"]
            bindings = self.bindings.get((group, scope), {})
            if root["work"]["bindings"] != bindings or contents.invocation_bindings(0, group) != bindings:
                raise ValueError("instantiated graph changes its native invocation binding")
            self.counts["bound_invocations"] += bool(bindings)
            self.counts["action_graphs"] += 1
            for boundary in self.ports[group, scope]:
                target = contents.boundary_event(0, boundary["at"])
                if target is None:
                    raise ValueError(f"required native resource tail is absent at {group}, scope {scope}: {boundary['at']}")
                event = row["event_ids"][target]
                self.boundaries[event][self.ranks[group], boundary["event"]] = None
            self.measured_graph = (self.ranks[group], scope, contents, row["event_ids"], {}, row["work_ns"])
            return scope

        call = self.cpu_calls[group, tuple(record["source"]["post"]), record["source"]["phase"]]
        stamps = self.case.cpu.native_times
        self.boundaries[ports["call.enter", "input"]][None, stamps[call.entered]] = None
        self.boundaries[ports["call.return", "output"]][None, stamps[call.returned]] = None
        if call.posted is not None:
            predicted = tuple(root["bindings"]["cpu_protocol"]["key"])
            if self.cpu_messages.setdefault(predicted, call.key) != call.key:
                raise ValueError("CPU message pairs posts that belong to different native transfers")
            self.boundaries[ports["transport.post", "output"]][None, stamps[call.posted]] = None
        return None

    def check_host_tails(self, rank, root, target, event_ids, matches):
        source, _, _ = self.case.sources[rank]
        native = source.host_tails(root)
        predicted = {event_ids[event]: identity for event, identity in target.host_tails(0).items()}
        native_counts = Counter(native.values())
        predicted_counts = Counter(predicted.values())
        for event, (source_event, time_ns) in matches.items():
            identity = native.get(source_event)
            if (identity is None or identity != predicted.get(event)
                    or native_counts[identity] != 1 or predicted_counts[identity] != 1 or time_ns is None):
                raise ValueError(f"rank {rank} action {root} consumes an unmatched host completion")
            if source_event in self.case.logical[rank]:
                raise ValueError(f"rank {rank} action {root} times an unmeasured host completion")
            declared = time_ns + self.origin_ns
            self.realization.condition(rank, source_event, declared)
            self.boundaries[event][rank, source_event] = declared
            self.counts["matched_host_tails"] += 1

    def check_measured_graph(self, rank, root, target, event_ids, matches, durations):
        source, raw, _ = self.case.sources[rank]
        native = ActionBoundaries(source, root)
        predicted = ActionBoundaries(target, 0, event_ids)
        identities = {event: source_event for event, (source_event, _) in matches.items()}
        mapped_events = set(identities.values())
        if identities.keys() != set(event_ids) or len(mapped_events) != len(event_ids):
            raise ValueError(f"rank {rank} action {root} lacks one-to-one native event coverage")
        owned = {event for scope in source.paths(root) for event in source.events[scope]} | set(native.events.values())
        owned.update(event for _, begin, end in native.work for event in (begin, end))
        if mapped_events - owned:
            raise ValueError(f"rank {rank} action {root} consumed an unrelated native event")
        if {name: identities[event] for name, event in predicted.events.items()} != native.events:
            raise ValueError(f"rank {rank} action {root} changes native semantic ports")
        predicted_work = Counter((identity[:2], identities[begin], identities[end]) for identity, begin, end in predicted.work)
        if predicted_work != Counter((identity[:2], begin, end) for identity, begin, end in native.work):
            raise ValueError(f"rank {rank} action {root} changes native work endpoint pairs")

        # Certify the declared calibration even for conditions in an unfinished execution prefix.
        for event, (source_event, time_ns) in matches.items():
            if (time_ns is None) != (source_event in self.case.logical[rank]):
                raise ValueError(f"rank {rank} action {root} changes the timing evidence of event {source_event}")
            declared = None if time_ns is None else time_ns + self.origin_ns
            if declared is not None:
                self.realization.condition(rank, source_event, declared)
            self.boundaries[event][rank, source_event] = declared

        if rank not in self.service_pairs:
            self.service_pairs[rank] = self.case.measured_service_pairs(rank, source.graph, raw)
        for piece, duration in zip(target.graph["work"], durations, strict=True):
            (start, begin_ns), (finish, end_ns) = (matches[event_ids[piece[side]]] for side in ("start", "finish"))
            pair = start, finish
            if (duration is not None) != (pair in self.service_pairs[rank]):
                raise ValueError(f"rank {rank} action {root} changes isolated work service evidence")
            if duration is not None:
                if end_ns - begin_ns != duration:
                    raise ValueError(f"rank {rank} interval {start}..{finish} changes its isolated service duration")
                self.counts["isolated_observed_services"] += 1
        self.counts.update(full_graphs=1, full_events=len(matches), full_work=len(native.work))

    def finish(self):
        if self.expected is None:
            return {"status": "not_used", "verified_prefix": False,
                    "scope": "No consumed-condition witness; ordinary logical pairing only."}
        remaining = sum(len(records) - self.positions[group] for group, records in self.expected.items())
        complete = not remaining and not self.pending and not self.reads and not self.boundaries
        full = complete and not self.peer_handoffs and not self.counts["cpu"] and 0 < self.counts["full_graphs"] == self.counts["action"]
        if self.peer_handoffs:
            self.counts["shared_boundary_graphs"] = self.counts["action_graphs"] - self.counts["full_graphs"]
        clock = self.realization.solve()
        return {"status": "complete" if complete else "partial", "verified_prefix": True,
                "full_node_coverage": full,
                "checked": dict(self.counts), "unconsumed": remaining, "unbound_graphs": len(self.pending), "unreturned_reads": len(self.reads),
                "unfinished_boundaries": len(self.boundaries), "clock_realization": clock.report(),
                "scope": "Consumed native identities follow measured CPU order. Timed conditions use original measurements and one joint clock; logical witnesses leave completion to actual dependencies."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", type=Path, help="require every action endpoint from prologue action-layouts")
    args = parser.parse_args()
    report = NativeInputs(args.capture, args.graphs, args.output, args.layouts).run()
    raise SystemExit(0 if report["status"] == "extracted" else 1)


if __name__ == "__main__":
    main()
