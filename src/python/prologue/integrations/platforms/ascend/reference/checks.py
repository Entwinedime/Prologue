"""Check coarse graph effects, launches and boundaries against the independent recording."""

from collections import Counter, defaultdict

from prologue.integrations.platforms.ascend.reference.work_evidence import WorkEvidence
from prologue.integrations.platforms.ascend.trace_store import (
    Interval,
    ThreadIntervals,
)


class NativeEffectEvidence(WorkEvidence):
    """Check projected effects directly against the raw recording and report which native work has independent
    evidence.
    """

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
