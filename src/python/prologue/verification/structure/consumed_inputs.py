"""Check which independent timing conditions a predicted execution actually used."""

from collections import Counter, defaultdict
from itertools import pairwise

from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.core.action_order import ActionBoundaries
from prologue.verification.core.source_clock import ClockRealization


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

            # Original CPU entries establish order; graph scope references retain their independently checked
            # raw origins.
            host = {int(row["name"].split("/")[1]): row["begin_raw_ns"]
                    for row in raw["host_observation"]["intervals"] if row["name"].startswith("prologue.scope/")}
            host.update({query["scope"]: query["begin_raw_ns"] for query in raw["state_queries"]})
            for index, identity in identities.items():
                self.ports[group, index] = tuple(
                    self.case.completion_boundaries(rank, index, peer_handoffs=self.peer_handoffs)
                )

                if bindings := source.invocation_bindings(index, group):
                    self.bindings[group, index] = bindings
                origins = source.graph["scopes"][index]["bindings"]["source_scopes"]
                begin = min(
                    raw["scopes"][origin]["host_interval"]["begin_raw_ns"]
                    if "service" in raw["scopes"][origin]
                    else host[origin]
                    for origin in origins
                )
                ordered[group].append(
                    (
                        begin,
                        {
                            "operation": "action",
                            "identity": {key: identity[key] for key in ("action", "parameters", "work")},
                            "source": {"scope": index},
                        },
                    )
                )

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
                for index, (interval, read) in enumerate(
                    zip(raw["request_input_attempts"], self.ingress.native_reads[instance], strict=True)
                ):
                    ordered[group].append((interval["begin_raw_ns"], {
                        "operation": "input", "identity": read["identity"], "source": {"input": index},
                    }))

        self.expected = {}
        for group, records in ordered.items():
            records.sort(key=lambda item: item[0])
            if any(left[0] >= right[0] for left, right in pairwise(records)):
                raise ValueError("native condition witnesses need strictly ordered measured CPU entries")

            self.expected[group] = [record for _, record in records]

    def observe(self, row):
        kind = row["kind"]
        self.observe_measured_boundary(row)

        if kind == "condition_used":
            self.consume_condition(row)
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

    def observe_measured_boundary(self, row):
        """Collect and discharge the measured endpoint obligations of one graph."""

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


    def consume_condition(self, row):
        """Retain the native condition until the corresponding graph or input read identifies it."""

        if self.expected is None:
            self.prepare()
            if self.realization is None:
                self.realization = ClockRealization(self.case.clocks, self.case.graph, self.case.cpu.native_times)

        group = row["group"]
        position = self.positions[group]
        if (row.keys() != {"kind", "group", "step", "operation", "source"} or type(row["step"]) is not int
                or row["step"] != position or position >= len(self.expected[group]) or group in self.pending):
            raise ValueError(f"invalid consumed execution condition position: {group} at {position}")

        # The following graph or input read supplies the actual identity; the receipt only selects its
        # original condition.
        record = self.expected[group][position]
        if row["operation"] != record["operation"] or row["source"] != record["source"]:
            raise ValueError(
                f"consumed execution condition differs from its native causal prefix: {group} at {position}"
            )

        self.positions[group] += 1
        self.counts[record["operation"]] += 1
        if record["operation"] == "input":
            instance = record["identity"]["instance"]
            if instance in self.reads:
                raise ValueError("input condition was consumed before the preceding read returned")

            self.reads[instance] = record["source"]["input"], record["identity"]
        else:
            self.pending[group] = record

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
                    raise ValueError(
                        f"required native resource tail is absent at {group}, scope {scope}: {boundary['at']}"
                    )

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

        predicted_work = Counter(
            (identity[:2], identities[begin], identities[end]) for identity, begin, end in predicted.work
        )
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
        full = (
            complete
            and not self.peer_handoffs
            and not self.counts["cpu"]
            and 0 < self.counts["full_graphs"] == self.counts["action"]
        )

        if self.peer_handoffs:
            self.counts["shared_boundary_graphs"] = self.counts["action_graphs"] - self.counts["full_graphs"]
        clock = self.realization.solve()
        return {
            "status": "complete" if complete else "partial",
            "verified_prefix": True,
            "full_node_coverage": full,
            "checked": dict(self.counts),
            "unconsumed": remaining,
            "unbound_graphs": len(self.pending),
            "unreturned_reads": len(self.reads),
            "unfinished_boundaries": len(self.boundaries),
            "clock_realization": clock.report(),
            "scope": (
                "Consumed native identities follow measured CPU order. Timed conditions use original "
                "measurements and one joint clock; logical witnesses leave completion to actual dependencies."
            ),
        }
