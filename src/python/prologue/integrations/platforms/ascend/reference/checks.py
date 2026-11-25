"""Check coarse graph effects, launches and boundaries against the independent recording."""

from collections import Counter, defaultdict, deque
from functools import cached_property

from prologue.integrations.platforms.ascend.reference.recording import SdkRecording
from prologue.integrations.platforms.ascend.trace_store import (
    HOST_WAITS,
    Interval,
    ThreadIntervals,
    native_interval,
)


class StreamEffects:
    def __init__(self, recording: SdkRecording):
        self.recording = recording

    def inspect_elided_waits(self, graph):
        recording = self.recording
        returns = {point: call for point, call in recording.logical_coordinates.items() if call is not None}
        if not returns:
            return

        coordinates = graph["evidence"]["events"]
        predecessors = defaultdict(set)
        for edge in graph["edges"]:
            if edge["kind"] not in {"signal", "submission"}:
                continue
            source, target = edge["source"], edge["target"]
            position = graph["events"][target]["resource"], coordinates[target]["order"]
            if position in returns:
                resource = graph["resources"][graph["events"][source]["resource"]]["identity"]
                predecessors[position].add((edge["kind"], resource, coordinates[source]["order"]))

        _, waits = recording.event_generations
        signals = {(row["thread"], row["begin_counter"]): connection
                   for (_, connection), choices in waits.items() for row, _ in choices}
        for position, (thread, begin) in returns.items():

            # A logical return may publish readiness only after the actual submission and recorded signal.
            recording.require((thread, begin) in signals, "elided wait lacks a native event generation")
            event = recording.sdk.event(signals[thread, begin])
            expected = {
                ("submission", f"thread/{thread}", begin),
                ("signal", f"stream/{event[0]}", recording.sdk.span(event)[1]),
            }
            recording.require(
                expected <= predecessors[position], "elided wait loses its native submission or signal dependency"
            )

    def enclosing_copy_tasks(self, interval, api, owner, stream):
        recording = self.recording
        submitted = recording.queues.inside(interval, stream)
        candidates = []
        for parent in recording.api_ranges.overlapping(api) if submitted else []:
            group = recording.sdk.connections.get(parent.record["args"]["connection_id"], [])
            if (
                group
                and parent.contains(api)
                and recording.owner(parent) == owner
                and recording.queues.inside(parent, stream) == submitted
            ):
                candidates.append(group)
        recording.require(len(candidates) <= 1, "native copy has ambiguous enclosing device work")
        return candidates[0] if candidates else []

    def copy_completion(self, row, effect, copy, tasks):
        witness = effect["completion"]
        completed = None
        if copy and not tasks:
            self.recording.require(witness is not None, "unmatched copy has no raw completion witness")
            completed = self.recording.runtime_records[witness["thread"], witness["begin_counter"]]
            self.recording.require(
                completed["api"] == witness["api"]
                and completed["api"] in {"drvMemcpy", "halMemcpy"}
                and completed["thread"] == row["thread"]
                and completed["result"] == 0
                and row["begin_counter"]
                <= completed["begin_counter"]
                <= completed["end_counter"]
                <= row["end_counter"]
                and all(completed[key] == row[key] for key in ("source", "destination", "bytes")),
                "copy completion does not witness a synchronous transfer of the actual buffers",
            )
        else:
            self.recording.require(
                witness is None, "queued work or stream wait claims an unrelated host-copy completion"
            )
        return completed

    def inspect(self, graph):
        recording = self.recording
        effects = recording.facts["native_effects"]
        recorded = {(effect["thread"], effect["begin_counter"]): effect for effect in effects}
        recording.require(len(recorded) == len(effects), "native effects repeat an API instance")
        expected, coordinates, counts = set(), {}, Counter()
        for row in recording.runtime_records.values():
            if row["api"] == "aclrtMemcpy" and row["bytes"] > 0:
                interval = native_interval(row)
                if recording.window.begin <= interval.begin and interval.end <= recording.window.end:
                    owner = recording.owner(recording.native_api(interval))
                    if owner is not None and owner not in recording.closed_scopes:
                        self.inspect_transfer(owner, row, graph)
                        counts["synchronous_copies"] += 1
            copy = row["api"] == "aclrtMemcpyAsync" and row["bytes"] > 0
            if not copy and row["api"] not in recording.waits:
                continue
            interval = native_interval(row)
            if interval.end <= recording.window.begin or recording.window.end <= interval.begin:
                continue
            api = recording.native_api(interval)
            owner = recording.owner(api)
            if owner is None or owner in recording.closed_scopes:
                continue
            key = row["thread"], row["begin_counter"]
            expected.add(key)
            recording.require(key in recorded, f"native stream effect is omitted: {key}, {row['api']}")
            effect = recorded[key]
            connection, stream = api.record["args"]["connection_id"], row["stream"]
            recording.require(
                row["result"] == 0 and row["stream_status"] == 0,
                "raw stream operation failed or has no stream identity",
            )
            recording.require(
                (effect["source_scope"], effect["connection"]) == (owner, connection),
                "native effect changes its API or owner",
            )
            resource = recording.facts["resources"][effect["resource"]]
            tasks = recording.sdk.connections.get(connection, [])
            if copy and not tasks:
                tasks = self.enclosing_copy_tasks(interval, api, owner, stream)
                counts["enclosing_group_copies" if tasks else "host_completed_copies"] += 1
            completed = self.copy_completion(row, effect, copy, tasks)
            if completed is not None:
                host = self.inspect_transfer(owner, completed, graph)
                recording.require(effect["resource"] == host and not effect["commands"],
                             "synchronous transfer invents a stream command or changes its actual caller")
                counts["copies"] += 1
                continue
            recording.require(
                (resource["kind"], resource["identity"]) == ("stream", f"stream/{stream}"),
                "native effect changes stream",
            )
            recording.require(all(task[0] == stream for task in tasks), "native effect's SDK tasks change stream")
            instances = defaultdict(deque)
            for task in tasks:
                instances[recording.sdk.key(recording.sdk.records[task]["args"])].append(task)
            expected_tasks = (
                Counter({key: len(values) for key, values in instances.items()}) if tasks else Counter({None: 1})
            )
            recording.require(
                Counter(
                    recording.sdk.key(command["sdk"]) if command["sdk"] is not None else None
                    for command in effect["commands"]
                )
                == expected_tasks,
                "native effect duplicates, omits or invents device work",
            )

            if not copy and not tasks:
                recording.require(
                    all(submission.record["tasks"] == 0 for submission in recording.queues.inside(interval, stream)),
                    "unmatched stream wait has actual driver submissions",
                )
                command, = effect["commands"]
                recording.logical_coordinates[effect["resource"], command["begin"]] = None
                recording.logical_coordinates[effect["resource"], command["end"]] = key
                counts["elided_stream_waits"] += 1

            for command in effect["commands"]:
                begin, end = command["begin"], command["end"]
                interval = Interval(begin, end, effect["resource"], {})
                recording.require(begin < end and recording.span_ranges.enclosing(interval) is not None,
                             "native effect has no retained stream span")
                if copy:
                    recording.require(
                        recording.payload_ranges.enclosing(interval) is not None,
                        "native copy is absent from work payload",
                    )
                else:
                    recording.require(
                        (effect["resource"], begin, end) in recording.wait_intervals, "native stream wait is absent"
                    )
                if command["sdk"] is not None:
                    identity = instances[recording.sdk.key(command["sdk"])].popleft()
                    recording.require(
                        coordinates.setdefault(identity, (begin, end)) == (begin, end),
                        "nested API views duplicate a device task",
                    )
            counts["copies" if copy else "stream_waits"] += 1
        recording.require(set(recorded) == expected, "effect evidence contains an unobserved or auxiliary operation")
        counts["device_measurements"] = recording.sdk.inspect_coordinates(recording.facts, recording.apis, coordinates)
        self.inspect_elided_waits(graph)
        return dict(counts)

    def inspect_transfer(self, source_scope, completed, graph):
        recording = self.recording
        scope = recording.source_scopes[source_scope]
        resources = recording.facts["resources"]
        host, = [index for index, resource in enumerate(resources)
                 if (resource["kind"], resource["identity"]) == ("thread", f"thread/{completed['thread']}")]
        begin, end = completed["begin_counter"], completed["end_counter"]
        recording.require(recording.spans[scope, host, begin, end, "transfer"] > 0,
                     "synchronous copy loses its caller transfer interval")
        recording.require(
            begin < end and recording.payload_ranges.enclosing(Interval(begin, end, host, {})) is not None,
            "synchronous transfer is absent from its caller payload",
        )
        raw = recording.native["scopes"][source_scope]

        # Only a semantic device owner publishes copy readiness; caller envelopes contain observer markers.
        while ("device_entry" not in raw or raw.get("envelope", False)) and raw["parent"] is not None:
            raw = recording.native["scopes"][raw["parent"]]
        if "device_entry" not in raw:
            return host

        owner = recording.source_scopes[raw["id"]]
        port, = [port for port in recording.facts["scopes"][owner]["ports"] if port["name"] == "device.complete"]
        published, = [port for port in graph["scopes"][owner]["ports"] if port["name"] == "device.complete"]
        recording.check_device_end(graph, published["event"], recording.point_connection(raw["device_return"]))
        recording.require((tuple(sorted(port["at"].items())), host, end) in recording.transfer_signals,
                     "synchronous transfer has no readiness edge to its actual enclosing return fence")

        return host


class NativeEffectEvidence(SdkRecording):
    @staticmethod
    def peer_exchange(scope):
        participants = scope.get("communication")
        return participants is not None and (
            "peer_rows" in scope or participants["required"] == participants["members"]
        )

    def inspect_host_fences(self, graph):
        resources = graph["resources"]
        locations = {
            (node["resource"], point["order"]): event
            for event, (node, point) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True))
            if resources[node["resource"]]["kind"] == "thread"
        }
        fences = {}
        for scope, lane, begin, end, kind in self.spans:
            resource = resources[lane]
            if kind != "wait" or resource["kind"] != "thread":
                continue
            source, = self.facts["scopes"][scope]["bindings"]["source_scopes"]
            raw = self.native["scopes"][source]
            if "service" in raw:
                interval = self.scopes[source].record
                if (begin, end) == (interval["begin_counter"], interval["end_counter"]):
                    continue

            tid = int(resource["identity"].split("/")[1])
            call = self.runtime_records.get((tid, begin))
            self.require(call is not None and call["api"] in HOST_WAITS
                         and call["end_counter"] == end and call["result"] == 0,
                         "recorded host fence must use its native runtime entry and return")
            api = self.native_api(native_interval(call))
            self.require(self.owner(api) == source, "host fence changes its native owner")
            fences[locations[lane, begin], locations[lane, end]] = call

        for scope in graph["scopes"]:
            for port in scope["ports"]:
                if "wait" not in port:
                    continue
                wait = port["wait"]
                call = fences[wait["begin"], port["event"]]
                if call["api"] != "aclrtSynchronizeDeviceWithTimeout":
                    stream = resources[wait["resource"]]
                    self.require(call["stream_status"] == 0 and stream["identity"] == f"stream/{call['stream']}",
                                 "host fence changes its recorded execution stream")
        return frozenset(fences)

    def replay_host_intervals(self):
        observer = [self.host_interval(interval) for interval in self.excluded]
        dequeues = []
        for scope, read in self.queries.items():
            wrapper = self.scopes[scope].record
            observer.append((wrapper["thread"], wrapper["begin_raw_ns"], read["begin_raw_ns"]))
            observer.append((wrapper["thread"], read["end_raw_ns"], wrapper["end_raw_ns"]))

        # Queue entries owned by the observer can use a different native thread.
        for interval, observer_owned in self.dequeued_intervals():
            dequeues.append(interval)
            if observer_owned:
                observer.append(interval)
        repeated = set(map(tuple, self.facts["measurements"].get("repeated_host_waits", ())))
        waits = list(self.reserved_host_intervals(repeated))
        for lane, begin, end in self.wait_intervals:
            resource = self.facts["resources"][lane]
            if resource["kind"] == "thread":
                identity = resource["identity"]
                thread = int(identity.split("/")[1])
                waits.append((thread, self.host_times[identity, begin], self.host_times[identity, end]))
        return observer, dequeues, waits

    def inspect_measurements(self, external):
        checked = 0
        unmeasured = set()
        observed = defaultdict(set)
        for point in self.facts["measurements"]["coordinates"]:
            resource = self.facts["resources"][point["resource"]]
            lane = resource["identity"], resource["role"], resource["serial"]
            if lane in external:
                self.require(resource["kind"] == "service" and point["time_ns"] is None,
                             "single-device operand staging claims target transport time")
                observed[lane].add(point["order"])
                continue
            if resource["kind"] == "stream":
                if point["time_ns"] is None:
                    unmeasured.add((point["resource"], point["order"]))
                continue
            key = resource["identity"], point["order"]
            self.require(key in self.host_times, f"host boundary has no original time source: {key}")
            self.require(
                point["time_ns"] == self.host_times[key],
                f"host measurement is missing or changes its original time: {key}",
            )
            checked += 1

        # inspect() independently checks these native commands against raw API and SDK task records.
        native_only = {(effect["resource"], command[side])
                       for effect in self.facts["native_effects"] for command in effect["commands"]
                       if command["sdk"] is None for side in ("begin", "end")}
        self.require(observed == external, "external service boundaries differ from original posts and waits")
        self.require(
            unmeasured <= native_only, "device boundary loses a recorded timestamp or lacks an observed native command"
        )
        logical = self.logical_coordinates.keys()
        return {"checked_host_measurements": checked, "external_transport_endpoints": sum(map(len, observed.values())),
                "unmeasured_device_endpoints": len(unmeasured - logical),
                "logical_wait_boundaries": len(unmeasured & logical)}

    def device_intervals(self):
        intervals = {}
        for scope in self.native["scopes"]:
            if "device_entry" not in scope:
                continue
            endpoints = []
            for field, side in (("device_entry", "entry"), ("device_return", "return")):
                point = scope[field]
                self.require(self.native["points"][point] == {"scope": scope["id"], "side": side},
                             "device interval refers to another call's original marker")
                endpoints.append(self.sdk.event(self.point_connection(point)))
            first, last = endpoints
            self.require(first[0] == last[0], "native device interval changes streams")
            begin, end = self.sdk.span(first)[1], self.sdk.span(last)[1]
            self.require(begin <= end, "native device interval reverses its original markers")
            intervals[scope["id"]] = Interval(begin, end, first[0], {})
        return intervals

    def inspect_boundaries(self, graph, device_intervals):
        spans = self.spans
        resources = {
            (resource["kind"], resource["identity"]): index for index, resource in enumerate(graph["resources"])
        }
        self.require(len(resources) == len(graph["resources"]), "coarse graph repeats a native resource")

        # Pure CPU declarations still require independent proof that they own no device payload.
        device_owners = {
            self.owner(api) for connection, api in self.apis.items() if self.sdk.connections.get(connection)
        }
        counts = Counter()
        for index, scope in enumerate(graph["scopes"]):
            origin, = scope["bindings"]["source_scopes"]
            if origin == 0:
                continue
            raw = self.native["scopes"][origin]
            thread = raw["thread"]
            begin = end = self.scopes[origin].record
            self.require(begin["thread"] == thread, f"scope {index} changes its native caller thread")
            ports = {port["name"]: port for port in scope["ports"]}
            kind = "wait" if "service" in raw else "boundary" if raw.get("envelope", False) else "host"
            if "communication" in raw and "device_entry" not in raw:
                kind = "collective" if self.peer_exchange(raw) else "communication"
            host = resources["thread", f"thread/{thread}"]
            if origin in self.queries:
                self.require(spans[index, host, begin["begin_counter"], end["end_counter"], "boundary"] == 1,
                             f"query {index} loses its uncharged caller envelope")
                read = self.queries[origin]
                self.require(
                    begin["begin_counter"] <= read["begin_counter"] <= read["end_counter"] <= end["end_counter"],
                    f"query {index} escapes its original caller interval",
                )
                self.check_boundary(graph, ports["caller.return"]["event"], host, end, "end", index)
                counts["caller_boundaries"] += 1
                begin = end = read
                kind = "host"
            self.require(spans[index, host, begin["begin_counter"], end["end_counter"], kind] == 1,
                         f"scope {index} omits, duplicates or changes its native host interval")
            for name, side, record in (("call.enter", "begin", begin), ("call.return", "end", end)):
                self.check_boundary(graph, ports[name]["event"], host, record, side, index)
                counts["host_boundaries"] += 1
            counts["host_intervals"] += 1
            if raw["id"] in device_intervals:
                interval = device_intervals[raw["id"]]
                endpoints = [ports[name]["event"] for name in ("device.enter", "device.complete") if name in ports]
                self.require(len(endpoints) in {0, 2}, "device interval loses one public endpoint")
                if endpoints:
                    stream = resources["stream", f"stream/{interval.thread}"]
                    self.require(all(graph["events"][event]["resource"] == stream for event in endpoints),
                                 "device interval changes its original stream")
                    first, last = (graph["evidence"]["events"][event]["order"] for event in endpoints)

                    # Semantic trimming may remove observer-only tails, but cannot escape the original markers.
                    self.require(interval.begin <= first <= last <= interval.end,
                                 "semantic device interval escapes its original marker envelope")
                    for event, original, position, point in zip(
                        endpoints, (interval.begin, interval.end), (first, last),
                        (raw["device_entry"], raw["device_return"]), strict=True,
                    ):

                        # An unconsumed recorder command only witnesses when its semantic port was already ready.
                        if original == position and point not in self.queried_points:
                            self.observer_joins.add(event)
                    counts["device_intervals"] += 1
            if scope["role"] == "segment" and raw["bindings"].get("segment", {}).get("host_only", False):
                self.require(origin not in device_owners, f"CPU-only scope {index} hides native device work")
                counts["cpu_only_regions"] += 1
            if "service" in raw:
                service = resources["service", f"service/{thread}"]
                self.require(
                    not graph["resources"][service]["serial"],
                    "synchronous service is charged as a serial caller resource",
                )
                self.require(
                    spans[index, service, begin["begin_counter"], end["end_counter"], raw["service"]["kind"]] == 1,
                    f"scope {index} omits, duplicates or changes its native service interval",
                )
                self.check_boundary(graph, ports["service.complete"]["event"], service, end, "end", index)
                counts["service_intervals"] += 1
            if "native_transfer" in raw:

                # Each payload keeps its own readiness interval; merging KV and later metadata invents early reads.
                for access in scope["kv_accesses"] + scope["slot_accesses"]:
                    self.check_boundary(graph, access["start"], service, begin, "begin", index)
                    self.check_boundary(graph, access["finish"], service, end, "end", index)
                counts["derived_state_accesses"] += len(scope["kv_accesses"]) + len(scope["slot_accesses"])
            else:
                counts["declared_state_accesses"] += len(raw.get("state_accesses", ()))
        return dict(counts)

    def inspect_services(self):
        services = {scope["id"]: scope for scope in self.native["scopes"] if "service" in scope}
        expected = Counter()
        observed = set()
        queues = {}

        # Reconstruct ownership from original submissions, without reading the producer's task assignments.
        for stream, post in self.queues.submissions:
            row = post.record
            if row["service"] < 0:
                continue
            scope = services[row["service"]]
            matches = [index for index, call in enumerate(scope["service"]["executions"])
                       if call["thread"] == row["thread"] and call["begin_counter"] <= row["begin_counter"]
                       and row["end_counter"] <= call["end_counter"]]
            self.require(len(matches) == 1, "private submission has no unique declared synchronous execution")
            previous = queues.get(stream)
            self.require(previous is None or previous[0] == scope["thread"] and previous[1] <= row["begin_counter"],
                         "closed service queue crosses callers or has concurrent submissions")
            queues[stream] = scope["thread"], row["end_counter"]
            expected[stream] += row["tasks"]
            observed.add((scope["id"], matches[0]))

        declared = {
            (scope["id"], index) for scope in services.values() for index in range(len(scope["service"]["executions"]))
        }
        self.require(observed == declared, "synchronous service executions lack raw queue evidence")
        self.require(
            not any(stream in expected and post.record["service"] < 0 for stream, post in self.queues.submissions),
            "private queue has unowned work outside its closed services",
        )
        for stream, count in expected.items():
            self.require(self.sdk.counts[stream] == count,
                         "private queue SDK coverage differs from its actual submitted SQE counts")

        self.require(
            not any(row["api"].startswith("aclrt") and row["stream_status"] == 0 and row["stream"] in expected
                    for row in self.runtime_records.values()),
            "closed service queue has external ACL operations",
        )
        return ({"synchronous_services": len(services), "executions": len(declared), "private_queues": len(expected),
                 "submitted_tasks": sum(expected.values())}, set(expected))

    @cached_property
    def staged_waits(self):
        first = {}
        for scope in self.native["scopes"]:
            for port in scope["ports"]:
                if port["direction"] != "input" or "producer" not in port:
                    continue
                producer = port["producer"]
                if self.native["scopes"][producer].get("local_exchanges"):
                    returned = scope["device_return"]
                    first[producer] = min(first.get(producer, returned), returned)
        return first

    def supplied_protocols(self, graph):
        expected = Counter()
        posts, handoffs = set(), set()
        external, ordered_posts = defaultdict(set), defaultdict(list)
        for scope in self.native["scopes"]:
            for call in scope.get("local_exchanges", ()):
                endpoints = []
                for field, side in (("begin", "exchange_begin"), ("end", "exchange_end")):
                    point = call[field]
                    self.require(self.native["points"][point] == {"scope": scope["id"], "side": side},
                                 "supplied operand refers to another protocol marker")
                    endpoints.append(self.sdk.event(self.point_connection(point)))
                first, last = endpoints
                self.require(first[0] == last[0] and first[1] < last[1], "protocol markers disagree on stream order")
                begin, end = self.sdk.span(first)[1], self.sdk.span(last)[1]
                exchange = scope["communication"]
                kind = "collective" if self.peer_exchange(scope) else "communication"
                lane = f"stream/{first[0]}"
                operand = lane, begin
                submitted = operand
                caller = self.points[call["begin"]]
                posting = f"thread/{caller.thread}", caller.begin
                if scope.get("asynchronous"):
                    self.require(
                        scope["id"] in self.staged_waits, "staged peer operand has no original consuming Work.wait"
                    )
                    payload, = [p for p in scope["ports"] if p["name"] == "payload"]
                    sends = exchange["rank"] == exchange["members"][0]
                    self.require(
                        exchange["operation"] == "transfer"
                        and call["operation"] == ("isend" if sends else "irecv")
                        and payload["direction"] == ("input" if sends else "output")
                        and call["dtype"] == payload["tensor"]["dtype"],
                        "staged operand differs from the native transfer port",
                    )
                    lane, begin, end = "service/peer", call["begin"], self.staged_waits[scope["id"]]
                    external[lane, "peer_transport", False].update((begin, end))
                elif kind == "collective":
                    service = f"service/{exchange['channel']}"
                    handoffs.add(("signal", service, call["end"], lane, end))
                    handoffs.add(("control", *posting, *operand))
                    submitted = lane, end
                    lane, begin, end = service, call["begin"], call["end"]
                    external[lane, "peer_collective", True].update((begin, end))
                    ordered_posts[lane].append((begin, caller.thread, caller.begin))
                expected[self.source_scopes[scope["id"]], kind, lane, begin, end,
                         call["operation"], call["dtype"], call["bytes"]] += 1
                posts.add((*posting, *submitted))
                if lane != operand[0]:
                    handoffs.add(("data", *operand, lane, begin))

        for lane, entries in ordered_posts.items():
            starts = [start for _, _, start in sorted(entries)]
            self.require(
                len({thread for _, thread, _ in entries}) == 1
                and all(a < b for a, b in zip(starts, starts[1:], strict=False)),
                f"collective service has no single native posting order on {lane}",
            )
        projected = Counter()
        for span in self.facts["spans"]:
            if operand := span.get("protocol_operand"):
                resource = graph["resources"][span["resource"]]
                projected[span["scope"], span["kind"], resource["identity"], span["begin"], span["end"],
                          operand["operation"], operand["dtype"], operand["bytes"]] += 1
        self.require(
            projected == expected, "projected protocol work differs from the original operand and stream markers"
        )
        self.require(len(posts) == expected.total(), "supplied operands repeat a native handoff")
        if handoffs:
            retained = set()
            for edge in self.facts["dependencies"]:
                if edge["kind"] not in {"data", "signal", "control"}:
                    continue
                source, target = edge["source"], edge["target"]
                resources = (self.facts["spans"][point["span"]]["resource"] for point in (source, target))
                origin, device = (self.facts["resources"][lane]["identity"] for lane in resources)
                retained.add((edge["kind"], origin, source["order"], device, target["order"]))
            self.require(handoffs <= retained, "staged service loses its caller-stream handoff")
        return posts, external

    def inspect_payload(self, graph, private_streams, device_intervals, submissions, launches=None, payload_times=None):
        expected, counts = defaultdict(list), Counter()
        envelopes = ThreadIntervals(list(device_intervals.values()))
        work = ThreadIntervals(
            [
                Interval(
                    graph["evidence"]["events"][item["start"]]["order"],
                    graph["evidence"]["events"][item["finish"]]["order"],
                    int(graph["resources"][item["resource"]]["identity"].split("/")[1]),
                    item,
                )
                for item in graph["work"]
                if graph["resources"][item["resource"]]["kind"] == "stream"
            ]
        )
        for identity, row in self.sdk.records.items():
            if row["args"]["Task Type"] in self.sdk.boundaries:
                continue
            stream = identity[0]
            if stream in private_streams:
                counts["opaque_service_tasks"] += 1
                continue
            api = self.apis.get(row["args"]["connection_id"])
            owner = self.owner(api) if api is not None else None
            begin, end = self.sdk.span(identity)
            interval = Interval(begin, end, stream, {})
            if api is not None:
                origin = self.origin(api)
                observer = self.excluded_ranges.enclosing(origin) is not None
                if observer or origin.end <= self.window.begin or self.window.end <= origin.begin:

                    # Fixed replay may need an excluded command's physical release and completion.
                    if observer and launches is not None:
                        self.sdk.observer_payload.add(identity)
                    counts["outside_scope_tasks"] += 1
                    continue
            if owner is None:
                self.require(envelopes.enclosing(interval) is not None,
                             "uncorrelated SDK payload lacks original semantic stream coverage")
            else:
                self.require(owner not in self.closed_scopes, "closed service payload lacks its private queue evidence")
            piece = work.enclosing(interval)
            self.require(piece is not None, "original SDK payload is absent from charged coarse work")
            if payload_times is not None:
                bounds = piece.record["start"], piece.record["finish"]
                payload_times[bounds].append(self.clock.interval(row))
            closed = piece.record["kind"] == "collective"
            if not closed or begin == piece.begin:
                post = self.check_payload_release(api, row["args"], stream, piece.begin, submissions)
                counts["checked_payload_releases"] += 1
                if launches is not None and begin == piece.begin:
                    launches[identity] = post, begin
            expected[stream].append([begin, end])
            counts["retained_payload_tasks"] += 1

        observed = {}
        for resource in self.facts["resources"]:
            if resource["kind"] == "stream":
                stream = int(resource["identity"].split("/")[1])
                observed[stream] = resource["payload"]
        self.require(
            all(observed.get(stream) == payload for stream, payload in expected.items()),
            "SDK payload is missing or changes its stream extent",
        )
        self.require(
            all(payload == expected.get(stream, []) for stream, payload in observed.items()),
            "stream payload includes unobserved or excluded SDK work",
        )
        return dict(counts)

    def check_payload_release(self, api, task, stream, entry, submissions):
        self.require(api is not None, "device payload has no raw post identity")
        post = self.queues.release(task)
        self.require((f"thread/{post.thread}", post.begin, f"stream/{stream}", entry) in submissions,
                     "device work can start before its original driver release")
        take = self.dequeue_ranges.enclosing(api)
        if take is not None:
            enqueue = self.enqueue[take.record["args"]["correlation_id"]]
            self.require(
                (f"thread/{enqueue.thread}", enqueue.begin, f"thread/{take.thread}", take.begin) in submissions,
                "submission thread can take work before the original enqueue",
            )
        return post

    def inspect_submissions(self, protocol_posts, launches=None):
        counts, links = Counter(), set()
        for edge in self.facts["dependencies"]:
            if edge["kind"] != "submission":
                continue
            source, target = edge["source"], edge["target"]
            resources = [
                self.facts["resources"][self.facts["spans"][point["span"]]["resource"]] for point in (source, target)
            ]
            link = resources[0]["identity"], source["order"], resources[1]["identity"], target["order"]
            links.add(link)
            if link in protocol_posts:
                continue
            if (resources[0]["kind"], resources[1]["kind"]) != ("thread", "stream"):
                continue
            thread = int(resources[0]["identity"].split("/")[1])
            stream = int(resources[1]["identity"].split("/")[1])
            resource = self.facts["spans"][target["span"]]["resource"]
            if self.logical_coordinates.get((resource, target["order"])) == (thread, source["order"]):
                counts["logical_waits"] += 1
                continue
            task = self.sdk.task(stream, target["order"])
            self.require(task is not None and task["args"]["connection_id"] in self.apis,
                         "device submission has no observed physical task")
            post = self.queues.release(task["args"])
            self.require((thread, source["order"]) == (post.thread, post.begin),
                         "device submission changes the actual driver post for its SDK task")
            task_kind = task["args"]["Task Type"]
            if launches is not None and task_kind in {"EVENT_WAIT", "EVENT_RECORD"}:
                identity, = self.sdk.connections[task["args"]["connection_id"]]

                # Retain the exact endpoint of the graph's submission edge.
                launches[identity] = post, target["order"]
            counts["driver_boundaries"] += 1
        self.require(protocol_posts <= links, "supplied protocol work lacks its original caller handoff")
        counts["supplied_protocol_operands"] = len(protocol_posts)
        return dict(counts), links

    def check_boundary(self, graph, event, resource, record, side, scope):
        measured = graph["evidence"]["events"][event]
        self.require(
            graph["events"][event]["resource"] == resource
            and measured["order"] == record[side + "_counter"]
            and measured["time_ns"] == record[side + "_raw_ns"],
            f"scope {scope} changes its native {side} boundary or measurement",
        )

    def inspect_stream_roles(self, points, declarations, waits):
        bindings = []
        for point, role in points.items():
            task = self.sdk.event(self.point_connection(point))
            bindings.append((task[0], role))

        observed = set()
        for (owner, connection), records in waits.items():
            if owner not in declarations:
                continue
            task = self.sdk.event(connection)
            for row, _ in records:
                if row["api"] in self.waits:
                    self.require(row["stream_status"] == 0, "named stream wait lacks its actual destination")
                    bindings.extend(
                        ((task[0], declarations[owner]["source"]), (row["stream"], declarations[owner]["target"]))
                    )
                    observed.add(owner)
        self.require(observed == declarations.keys(), "logical stream wait lacks original native evidence")

        # Resolve declarations from raw SDK/API generations independently of the producer's resource table.
        named, resources = {}, {}
        for stream, role in bindings:
            self.require(named.setdefault(role, stream) == stream and resources.setdefault(stream, role) == role,
                         "raw observations disagree on a logical stream binding")
        for resource in self.facts["resources"]:
            if resource["kind"] == "stream":
                stream = int(resource["identity"].split("/")[1])
                self.require(
                    resource["role"] == resources.get(stream, "execution"),
                    "stream loses or invents its declared logical role",
                )
        return {
            "named_streams": len(named),
            "point_declarations": len(points),
            "stream_wait_declarations": len(declarations),
        }

    def wait_matches(self, graph, event, choices, effects):
        resource = graph["events"][event]["resource"]
        position = graph["evidence"]["events"][event]["order"]
        for row, _ in choices:
            if row["api"] in self.waits:
                effect = effects[row["thread"], row["begin_counter"]]
                command, = effect["commands"]
                if resource == effect["resource"] and position == command["end"]:
                    return True
            else:
                descriptor = graph["resources"][resource]
                if (descriptor["kind"], descriptor["identity"]) == (
                    "thread",
                    f"thread/{row['thread']}",
                ) and position == row["end_counter"]:
                    return True
        return False

    def inspect_signals(self, contents, dependencies, generations):
        graph, counts = contents.graph, Counter()
        work, waits = generations

        producers, consumers = defaultdict(set), []
        effects = {(effect["thread"], effect["begin_counter"]): effect for effect in self.facts["native_effects"]}
        for raw in self.native["scopes"][1:]:
            index = contents.source_ids[raw["id"]]
            ports = {(port["name"], port["direction"]): port for port in graph["scopes"][index]["ports"]}
            for port in raw["ports"]:
                if port["use"] != "signal":
                    continue
                observed = ports[port["name"], port["direction"]]
                self.require(
                    observed["use"] == "signal" and observed["relation"] == "signal",
                    "native signal loses its port contract",
                )
                event = observed["event"]
                if "boundary" in port:
                    boundary = port["boundary"]
                    reference = graph["scopes"][contents.source_ids[boundary["scope"]]]
                    names = {
                        ("call", "begin"): "call.enter",
                        ("call", "end"): "call.return",
                        ("device", "begin"): "device.enter",
                        ("device", "end"): "device.complete",
                        ("service", "end"): "service.complete",
                    }
                    name = names[boundary["interval"], boundary["side"]]
                    reference, = [candidate for candidate in reference["ports"] if candidate["name"] == name]
                    self.require(event == reference["event"], "native boundary signal changes its referenced endpoint")
                    counts["boundary_signals"] += 1
                    continue
                if (producer := port.get("producer")) in self.staged_waits:
                    identity = f"work/{producer}"
                    self.require(observed["signal"] == identity, "staged signal changes its original Work handle")
                    if port["direction"] == "output":
                        self.require(event == ports["communication.complete", "output"]["event"],
                                     "staged Work publishes a different completion from its transfer")
                        producers[identity].add(event)
                    else:
                        entry = ports["device.enter", "input"]["event"]
                        self.require(event == ports["device.complete", "output"]["event"],
                                     "staged Work is consumed before the receiver's continuation")
                        self.check_device_end(graph, entry, self.point_connection(raw["device_entry"]))
                        self.check_device_end(graph, event, self.point_connection(raw["device_return"]))
                        self.require(entry != event and dependencies.reaches(entry, event)
                                     and dependencies.reaches(ports["call.enter", "input"]["event"], event),
                                     "staged wait completion loses its entry or caller prerequisite")
                        consumers.append((identity, event))
                    counts["staged_work_signals"] += 1
                    continue
                connection = self.point_connection(port["point"]) if "point" in port else work[port["producer"]]
                self.require(
                    observed["signal"] == f"event/{connection}", "native signal changes its completion generation"
                )
                if port["direction"] == "output":
                    self.check_device_end(graph, event, connection)
                    producers[connection].add(event)
                    counts["exported_completions"] += 1
                    continue

                # A wait consumes its actual generation at the CPU API return or stream-wait completion.
                self.require(self.wait_matches(graph, event, waits[raw["id"], connection], effects),
                             "native signal input is detached from its actual wait completion")
                consumers.append((connection, event))
                counts["signal_waits"] += 1

        for query in self.native["state_queries"]:
            if query["kind"] == "event" and query["result"]:
                scope = graph["scopes"][contents.source_ids[query["scope"]]]
                event, = [port["event"] for port in scope["ports"] if port["name"] == "call.return"]
                consumers.append((self.point_connection(query["point"]), event))
                counts["true_completion_reads"] += 1
        for connection, event in consumers:
            self.require(
                producers[connection]
                and any(dependencies.reaches(producer, event) for producer in producers[connection]),
                "native completion consumer has no path from its actual producer",
            )
        return dict(counts)
