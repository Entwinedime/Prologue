"""Bind device execution and collective service to recorded graph spans."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from prologue.integrations.platforms.ascend.analysis.commands import NativeStreamCall
from prologue.integrations.platforms.ascend.analysis.services import uses_peer_service
from prologue.integrations.platforms.ascend.analysis.trace import Interval

from .commands import span_references


class DeviceEvents:
    def __init__(self, graph):
        self.graph = graph

    def service_spans(self) -> None:
        graph = self.graph
        for scope, raw in enumerate(graph.raw["scopes"]):
            if "service" not in raw:
                continue
            interval = graph.trace.scopes[scope]
            service = raw["service"]
            resource = graph.resource("service", interval.thread, service["role"], serial=False)

            # A synchronous service joins its private workers; their incidental dispatch is not an external interface.
            span = graph.span(scope, resource, interval.begin, interval.end, service["kind"])
            if interval.begin < interval.end:
                graph.resources[resource]["payload"].append([interval.begin, interval.end])
            graph.services[scope] = span
            graph.edge(graph.endpoint(graph.host[scope], "begin"), graph.endpoint(span, "begin"), "submission")
            graph.edge(graph.endpoint(span, "end"), graph.endpoint(graph.host[scope], "end"), "signal")
            if "submitted_by" in raw:
                producer = graph.host[raw["submitted_by"]]
                graph.edge(graph.endpoint(producer, "end"), graph.endpoint(span, "begin"), "submission")

    def transfer_times(self) -> list:
        graph = self.graph
        samples = [
            [graph.spans[span]["resource"], interval.begin, interval.end,
             interval.record["end_raw_ns"] - interval.record["begin_raw_ns"], interval.record["bytes"], direction]
            for _, span, interval, direction in graph.host_copies
        ]
        for scope, identity in graph.services.items():
            raw = graph.raw["scopes"][scope]
            service = raw["service"]
            payload = service.get("payload")
            if not payload or not payload["bytes"] or not service["executions"]:
                continue
            span, stamps = graph.spans[identity], raw["host_interval"]

            # Queue attribution already checked the private submissions. Its synchronous return joins the workers.
            samples.append([span["resource"], span["begin"], span["end"], stamps["end_raw_ns"] - stamps["begin_raw_ns"],
                            payload["bytes"], payload["direction"]])
        return samples

    def device_spans(self) -> None:
        graph = self.graph
        points = graph.trace.device_points
        if set(points) != set(range(len(graph.raw["points"]))):
            raise ValueError("device point metadata and native markers differ")
        for scope, raw in enumerate(graph.raw["scopes"]):
            if "device_entry" not in raw:
                continue
            first, last = raw["device_entry"], raw["device_return"]
            for identity, side in ((first, "entry"), (last, "return")):
                if graph.raw["points"][identity] != {"scope": scope, "side": side}:
                    raise ValueError(f"scope {scope} refers to another call's device point")
            begin, end = points[first], points[last]
            if begin.stream != end.stream:
                raise ValueError(f"scope {scope} changes its boundary stream; explicit stream ports are required")
            resource = graph.resource("stream", begin.stream, "execution")
            kind = "boundary" if raw.get("envelope", False) else "communication" if "communication" in raw else "device"
            span = graph.span(
                scope, resource, graph.streams.task(begin.task).end, graph.streams.task(end.task).end, kind
            )
            graph.device[scope] = span

        # A native event may be recorded on another stream than its enclosing action's device interval.
        for identity, observed in enumerate(graph.raw["points"]):
            if observed["side"] != "signal":
                continue
            point = points[identity]
            resource = graph.resource("stream", point.stream, "execution")
            position = graph.streams.task(point.task).end
            graph.span(observed["scope"], resource, position, position, "boundary")

        graph.device_index = graph.span_index("stream")
        self.auxiliary_stream_spans()

        graph.device_index = graph.span_index("stream")
        for source in graph.streams.native_calls:
            if source.completion is None:
                continue

            # A successful driver transfer is caller work, not a queued device command.
            graph.host_transfer(source.scope, source.completion, source.native.record["direction"])
        for scope, transfer, _, _ in graph.host_copies:

            # An observation envelope cannot promote synchronous caller work into a device-stream fence.
            while scope != 0 and (scope not in graph.device or graph.raw["scopes"][scope].get("envelope", False)):
                scope = graph.scopes[scope]["parent"]
            if scope in graph.device:
                span = graph.device[scope]
                boundary = graph.spans[span]

                # Preserve the actual return fence as a join; never retarget CPU completion to an earlier device task.
                graph.span(scope, boundary["resource"], boundary["end"], boundary["end"], "boundary")
                graph.edge(graph.endpoint(transfer, "end"), graph.endpoint(span, "end"), "signal")
        for task in graph.streams.payload:
            source = task.source
            kind = source.record["args"]["Task Type"]
            if kind in {"EVENT_RECORD", "EVENT_WAIT", "PROFILING_ENABLE", "PROFILING_DISABLE"}:
                continue
            if source.service is not None or graph.trace.auxiliary(source) or graph.trace.outside_window(source):
                continue
            stream = task.stream
            resource = graph.resource("stream", stream, "execution")
            if graph.device_index.enclosing(Interval(task.begin, task.end, stream, {})) is None:
                raise ValueError(f"native payload has no semantic stream coverage: stream {stream} at {task.begin}")
            graph.payload.append(task)
            graph.resources[resource]["payload"].append([task.begin, task.end])

    def collective_spans(self) -> None:
        graph = self.graph
        groups = defaultdict(dict)
        for task in graph.payload:
            api = graph.trace.apis.get(task.source.record["args"]["connection_id"])
            if api is None:
                continue
            queue = graph.trace.dequeue_ranges.enclosing(api)
            if queue is None or not queue.record["name"].startswith("Dequeue@Hccl"):
                continue
            scope = graph.trace.owner(graph.trace.origin(api))
            if scope is None:
                continue
            raw = graph.raw["scopes"][scope]
            if not uses_peer_service(raw):
                continue
            if raw.get("asynchronous", False) and raw["communication"]["operation"] != "transfer":
                continue
            groups[scope].setdefault(queue.record["args"]["correlation_id"], []).append(task)

        spans = defaultdict(list)
        for span in graph.spans:
            if span["kind"] == "communication" and graph.resources[span["resource"]]["kind"] == "stream":
                spans[span["scope"]].append(span)
        first_wait = {}
        for waiter, producer in graph.waited_work.items():
            if graph.raw["scopes"][producer].get("local_exchanges"):
                returned = graph.raw["scopes"][waiter]["device_return"]
                first_wait[producer] = min(first_wait.get(producer, returned), returned)

        for scope, raw in enumerate(graph.raw["scopes"]):
            if "local_exchanges" not in raw:
                continue
            kind = "collective" if uses_peer_service(raw) else "communication"
            resource = graph.spans[graph.device[scope]]["resource"]
            for span in spans[scope]:
                span["kind"] = "device"
            for call in raw["local_exchanges"]:
                first, last = (graph.trace.device_points[call[key]] for key in ("begin", "end"))
                if first.stream != last.stream or graph.resource("stream", first.stream, "execution") != resource:
                    raise ValueError("a local protocol cut must keep its caller stream")
                posted = graph.host_anchor(first.host)
                command = None
                if raw.get("asynchronous"):
                    if scope not in first_wait or scope in graph.peer_services:
                        raise ValueError(f"staged Work {scope} needs one posted operand and a consuming wait")

                    # Post and wait identify the handle's lifetime. A target service can finish independently of the
                    # caller.
                    peer = graph.resource("service", "peer", "peer_transport", serial=False)
                    service = graph.span(scope, peer, call["begin"], first_wait[scope], kind)
                    graph.peer_services[scope] = service
                elif kind == "collective":

                    # HCCL uses the process group's queue for synchronous peer services, including exchanges with
                    # unequal splits.
                    lane = raw["communication"]["channel"]
                    peer = graph.resource("service", lane, "peer_collective")
                    service = graph.span(scope, peer, call["begin"], call["end"], kind)
                    command = graph.staged_boundary(scope, last)
                    graph.edge(graph.endpoint(service, "end"), command, "signal")
                else:
                    begin, end = graph.streams.task(first.task).end, graph.streams.task(last.task).end
                    service = graph.span(scope, resource, begin, end, kind)
                graph.spans[service]["protocol_operand"] = {key: call[key] for key in ("operation", "bytes", "dtype")}

                ready = graph.endpoint(service, "begin")
                if graph.spans[service]["resource"] != resource:
                    operand = graph.staged_boundary(scope, first)
                    graph.edge(operand, ready, "data")
                    ready = operand

                # One synchronous stream command is accepted at its caller cut and retires with the result.
                if command is not None:
                    graph.edge(posted, ready, "control")
                    graph.edge(posted, command, "submission")
                else:
                    graph.edge(posted, ready, "submission")
        for scope, calls in groups.items():
            if len(calls) != 1:
                continue
            (correlation, tasks), = calls.items()
            stream = tasks[0].stream
            if any(task.stream != stream for task in tasks):
                continue
            begin, end = tasks[0].begin, tasks[-1].end
            limit = graph.device_index.enclosing(Interval(begin, end, stream, {}))
            if limit is None:
                continue
            enclosing = graph.spans[limit.record["span"]]
            if enclosing["scope"] != scope or enclosing["kind"] != "communication":
                continue

            # Keep caller-stream work such as the gather transpose outside the peer-gated service.
            for span in spans[scope]:
                span["kind"] = "device"
            resource = enclosing["resource"]
            if (enclosing["begin"], enclosing["end"]) == (begin, end):
                enclosing["kind"] = "collective"
            else:
                graph.span(scope, resource, begin, end, "collective")
            graph.collectives[scope] = (resource, begin, end, graph.trace.dequeue[correlation])
            if graph.evidence and (signals := graph.trace.notifications.observed(tasks, graph.trace.clock)):
                graph.collective_notifications[str(scope)] = signals
        graph.device_index = graph.span_index("stream")

    def auxiliary_stream_spans(self) -> None:
        graph = self.graph
        pending = None
        communications = []
        for task in graph.streams.commands:
            source = task.source
            if isinstance(source, NativeStreamCall):
                resource = graph.resource("stream", task.stream, "execution")
                graph.span(source.scope, resource, task.begin, task.end, "wait")
                pending = None
                continue
            else:
                if source.service is not None or graph.trace.auxiliary(source) or graph.trace.outside_window(source):
                    continue
                api = graph.trace.apis.get(source.record["args"]["connection_id"])
                owner = None if api is None else graph.trace.owner(graph.trace.origin(api))
            stream = task.stream
            if graph.device_index.enclosing(Interval(task.begin, task.end, stream, {})) is not None:
                pending = None
                continue
            if owner is None:
                pending = None
                continue
            resource = graph.resource("stream", stream, "execution")
            key = owner, stream
            if pending is not None and pending[0] == key:
                graph.spans[pending[1]]["end"] = task.end
            else:
                communication = "communication" in graph.raw["scopes"][owner]
                span = graph.span(owner, resource, task.begin, task.end, "communication" if communication else "device")
                pending = key, span
                if communication:
                    communications.append((owner, span))

        # Other streams belong to the submitting scope; their actual event/wait records supply completion dependencies.
        for scope, span in communications:

            # Native tensor collectives publish usable output on their caller stream, even with internal communication
            # streams.
            caller = graph.device[scope]
            graph.edge(graph.endpoint(caller, "begin"), graph.endpoint(span, "begin"), "data")
            if not graph.raw["scopes"][scope].get("asynchronous", False):
                graph.edge(graph.endpoint(span, "end"), graph.endpoint(caller, "end"), "signal")

    def normalize_device_spans(self, result: dict) -> None:
        graph = self.graph
        anchors = list(span_references(result))
        references = Counter(anchor["span"] for anchor in anchors)
        device = set(graph.device.values())
        points = defaultdict(set)
        for identity, span in enumerate(graph.spans):
            if identity not in device and graph.resources[span["resource"]]["kind"] == "stream":
                points[span["resource"]].update((span["begin"], span["end"]))

        by_span = defaultdict(list)
        for anchor in {id(anchor): anchor for anchor in anchors}.values():
            identity, position = anchor["span"], anchor["order"]
            span = graph.spans[identity]
            by_span[identity].append(anchor)

            # Interval endpoints come from observer records; interior native dependencies and explicit points remain.
            if graph.resources[span["resource"]]["kind"] == "stream" and (
                identity not in device or span["begin"] < position < span["end"]
            ):
                points[span["resource"]].add(position)
        points = {resource: sorted(values) for resource, values in points.items()}
        for scope, span in graph.device.items():
            references[span] -= sum(
                port["name"] in {"device.enter", "device.complete"} for port in graph.scopes[scope]["ports"]
            )

        discarded = set()
        for identity in device:
            span = graph.spans[identity]
            begin, end = span["begin"], span["end"]
            positions = points.get(span["resource"], ())
            left, right = bisect_left(positions, begin), bisect_right(positions, end)
            bounds = [positions[left], positions[right - 1]] if left < right else []
            payload = graph.resources[span["resource"]]["payload"]
            first = bisect_right(payload, begin, key=lambda interval: interval[1])
            last = bisect_left(payload, end, key=lambda interval: interval[0])
            if first < last:
                bounds.extend((max(begin, payload[first][0]), min(end, payload[last - 1][1])))
            if not bounds:
                if references[identity] == 0:
                    discarded.add(identity)
                continue

            # Semantic device lifetime covers native work and required fences, not the observer's enclosing markers.
            span["begin"], span["end"] = min(bounds), max(bounds)
            if span["begin"] == span["end"]:
                span["kind"] = "boundary"
            for anchor in by_span[identity]:
                if anchor["order"] == begin:
                    anchor["order"] = span["begin"]
                elif anchor["order"] == end:
                    anchor["order"] = span["end"]
        if not discarded:
            return

        for scope in graph.scopes:
            scope["ports"][:] = [port for port in scope["ports"] if port["at"]["span"] not in discarded]

        # Publication ends interpretation: remap every surviving reference, including shared anchor objects, once.
        positions = {
            old: new for new, old in enumerate(index for index in range(len(graph.spans)) if index not in discarded)
        }
        anchors = {id(anchor): anchor for anchor in span_references(result)}
        for anchor in anchors.values():
            anchor["span"] = positions[anchor["span"]]
        result["spans"] = [span for index, span in enumerate(graph.spans) if index not in discarded]
