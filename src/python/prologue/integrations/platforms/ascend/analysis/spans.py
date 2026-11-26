"""Assign native intervals and work to resources with measured host and device coordinates."""

from __future__ import annotations

from collections import defaultdict

from prologue.integrations.platforms.ascend.analysis.commands import (
    NativeStreamCall,
    StreamCommand,
    StreamCommands,
)
from prologue.integrations.platforms.ascend.analysis.trace import (
    AscendTrace,
    DevicePoint,
    SynchronousCopy,
    uses_peer_service,
)
from prologue.integrations.platforms.ascend.trace_store import HOST_WAITS, Interval, ThreadIntervals


class ObservedSpans:
    """Separate chargeable work from host waits, scope envelopes, and private services."""

    def __init__(self, rank: int, raw: dict, trace: AscendTrace, *, evidence: bool) -> None:
        self.rank, self.raw, self.trace = rank, raw, trace
        self.streams = StreamCommands(trace)
        self.evidence = evidence
        self.control_threads = {
            interval.thread for scope, interval in self.trace.scopes.items() if "service" not in raw["scopes"][scope]
        }

        self.resources = []
        self.resource_ids = {}
        self.stream_names = {}
        self.spans = []
        self.dependencies = []

        self.host = {}
        self.callers = {}
        self.device = {}
        self.services = {}
        self.payload = []
        self.collectives = {}
        self.collective_notifications = {}
        self.host_copies = []
        self.work_completions = {}
        self.peer_services = {}
        self.staged_boundaries: dict[tuple[int, int], dict[str, int]] = {}

        self.waited_work = {
            scope["id"]: port["producer"]
            for scope in raw["scopes"]
            for port in scope["ports"]
            if port["use"] == "signal" and port["direction"] == "input" and "producer" in port
        }
        self.scopes = []

        for identity, scope in enumerate(raw["scopes"]):
            if scope["id"] != identity or not scope["complete"]:
                raise ValueError(f"native scope {identity} is not a completed, ordered call")
            info = {
                name: scope[name]
                for name in (
                    "parent",
                    "role",
                    "site",
                    "implementation",
                    "parameters",
                    "constants",
                    "arguments",
                    "work",
                    "bindings",
                )
            }
            info["bindings"] = dict(info["bindings"])

            info["guards"] = scope.get("guards", [])
            if "row_call" in scope:
                info["row_call"] = scope["row_call"]
            for name in ("effects", "publication"):
                if name in scope:
                    info["bindings"][name] = scope[name]

            # Raw and interpreted scope IDs have separate consumers; record their owner at production.
            info["bindings"]["source_scopes"] = [identity]
            if "communication" in scope:
                info["communication"] = scope["communication"]
                if "peer_rows" in scope:
                    info["communication"] = {**info["communication"], "peer_rows": scope["peer_rows"]}

            info["ports"] = []
            info["kv_accesses"] = []
            info["slot_accesses"] = []
            self.scopes.append(info)

    def resource(self, kind: str, identity: int | str, role: str, *, serial: bool = True) -> int:
        key = kind, identity
        if key not in self.resource_ids:
            if kind == "thread":
                role = "control" if identity in self.control_threads else "submission"
            elif kind == "stream":
                role = self.stream_names.get(identity, role)

            self.resource_ids[key] = len(self.resources)
            self.resources.append(
                {
                    "kind": kind,
                    "rank": self.rank,
                    "role": role,
                    "identity": f"{kind}/{identity}",
                    "serial": serial,
                    "observed": True,
                    "payload": [],
                }
            )

        return self.resource_ids[key]

    def span(self, scope: int, resource: int, begin: int, end: int, kind: str) -> int:
        if begin > end:
            raise ValueError(f"reversed native {kind} interval in scope {scope}")
        identity = len(self.spans)
        self.spans.append({"scope": scope, "resource": resource, "begin": begin, "end": end, "kind": kind})
        return identity

    @staticmethod
    def at(span: int, order: int) -> dict[str, int]:
        return {"span": span, "order": order}

    def endpoint(self, span: int, side: str) -> dict[str, int]:
        return self.at(span, self.spans[span][side])

    def edge(self, source: dict, target: dict, kind: str) -> None:
        if source != target:
            self.dependencies.append({"source": source, "target": target, "kind": kind})

    def host_anchor(self, interval: Interval, side: str = "begin") -> dict[str, int]:
        position = getattr(interval, side)
        boundary = self.host_index.enclosing(Interval(position, position, interval.thread, {}))
        if boundary is None:
            raise ValueError(f"required host endpoint has no observed scope: thread {interval.thread} at {position}")
        return self.at(boundary.record["span"], position)

    def span_index(self, kind: str) -> ThreadIntervals:
        intervals = []
        for identity, span in enumerate(self.spans):
            resource = self.resources[span["resource"]]
            if resource["kind"] == kind:
                coordinate = int(resource["identity"].split("/")[1])
                intervals.append(Interval(span["begin"], span["end"], coordinate, {"span": identity}))
        return ThreadIntervals(intervals)

    def host_transfer(self, scope: int, interval: Interval, direction: int, operand: dict | None = None) -> None:
        resource = self.resource("thread", interval.thread, "native")
        span = self.span(scope, resource, interval.begin, interval.end, "transfer")
        self.resources[resource]["payload"].append([interval.begin, interval.end])
        self.host_copies.append((scope, span, interval, direction))
        if operand is not None:
            self.spans[span]["copy"] = operand
            self.spans[span]["operand_bytes"] = interval.record["bytes"]

    def work_end(self, producer: int) -> dict:
        if producer in self.peer_services:
            return self.endpoint(self.peer_services[producer], "end")
        return self.task_anchor(self.work_completions[producer])

    def staged_boundary(self, scope: int, point: DevicePoint) -> dict[str, int]:

        # Retain the observed marker without charging a device operation.
        position = self.streams.task(point.task).end
        stream = self.resource("stream", point.stream, "execution")
        boundary = self.at(self.span(scope, stream, position, position, "boundary"), position)
        self.staged_boundaries[point.stream, position] = boundary
        return boundary

    def task_anchor(self, task: StreamCommand, side: str = "end") -> dict[str, int]:
        stream = task.stream
        position = getattr(task, side)
        if staged := self.staged_boundaries.get((stream, position)):
            return staged
        boundary = self.device_index.enclosing(Interval(position, position, stream, {}))
        if boundary is None:
            raise ValueError(f"required device endpoint has no semantic coverage: stream {stream} at {position}")
        return self.at(boundary.record["span"], position)

    def service_spans(self) -> None:
        for scope, raw in enumerate(self.raw["scopes"]):
            if "service" not in raw:
                continue
            interval = self.trace.scopes[scope]
            service = raw["service"]
            resource = self.resource("service", interval.thread, service["role"], serial=False)

            # A synchronous service joins its private workers; their incidental dispatch is not an external interface.
            span = self.span(scope, resource, interval.begin, interval.end, service["kind"])
            if interval.begin < interval.end:
                self.resources[resource]["payload"].append([interval.begin, interval.end])
            self.services[scope] = span
            self.edge(self.endpoint(self.host[scope], "begin"), self.endpoint(span, "begin"), "submission")
            self.edge(self.endpoint(span, "end"), self.endpoint(self.host[scope], "end"), "signal")

            if "submitted_by" in raw:
                producer = self.host[raw["submitted_by"]]
                self.edge(self.endpoint(producer, "end"), self.endpoint(span, "begin"), "submission")

    def transfer_times(self) -> list:
        samples = [
            [
                self.spans[span]["resource"],
                interval.begin,
                interval.end,
                interval.record["end_raw_ns"] - interval.record["begin_raw_ns"],
                interval.record["bytes"],
                direction,
            ]
            for _, span, interval, direction in self.host_copies
        ]

        for scope, identity in self.services.items():
            raw = self.raw["scopes"][scope]
            service = raw["service"]
            payload = service.get("payload")

            if not payload or not payload["bytes"] or not service["executions"]:
                continue
            span, stamps = self.spans[identity], raw["host_interval"]

            # Queue attribution already checked the private submissions. Its synchronous return joins the workers.
            samples.append(
                [
                    span["resource"],
                    span["begin"],
                    span["end"],
                    stamps["end_raw_ns"] - stamps["begin_raw_ns"],
                    payload["bytes"],
                    payload["direction"],
                ]
            )

        return samples

    def device_spans(self) -> None:
        points = self.trace.device_points
        if set(points) != set(range(len(self.raw["points"]))):
            raise ValueError("device point metadata and native markers differ")

        for scope, raw in enumerate(self.raw["scopes"]):
            if "device_entry" not in raw:
                continue
            first, last = raw["device_entry"], raw["device_return"]

            for identity, side in ((first, "entry"), (last, "return")):
                if self.raw["points"][identity] != {"scope": scope, "side": side}:
                    raise ValueError(f"scope {scope} refers to another call's device point")
            begin, end = points[first], points[last]

            if begin.stream != end.stream:
                raise ValueError(f"scope {scope} changes its boundary stream; explicit stream ports are required")
            resource = self.resource("stream", begin.stream, "execution")
            kind = "boundary" if raw.get("envelope", False) else "communication" if "communication" in raw else "device"
            span = self.span(scope, resource, self.streams.task(begin.task).end, self.streams.task(end.task).end, kind)
            self.device[scope] = span

        # A native event may be recorded on another stream than its enclosing action's device interval.
        for identity, observed in enumerate(self.raw["points"]):
            if observed["side"] != "signal":
                continue
            point = points[identity]
            resource = self.resource("stream", point.stream, "execution")
            position = self.streams.task(point.task).end
            self.span(observed["scope"], resource, position, position, "boundary")

        self.device_index = self.span_index("stream")
        self.auxiliary_stream_spans()

        self.device_index = self.span_index("stream")
        for source in self.streams.native_calls:
            if source.completion is None:
                continue

            # A successful driver transfer is caller work, not a queued device command.
            self.host_transfer(source.scope, source.completion, source.native.record["direction"])

        for scope, transfer, _, _ in self.host_copies:

            # An observation envelope cannot promote synchronous caller work into a device-stream fence.
            while scope != 0 and (scope not in self.device or self.raw["scopes"][scope].get("envelope", False)):
                scope = self.scopes[scope]["parent"]
            if scope in self.device:
                span = self.device[scope]
                boundary = self.spans[span]

                # Preserve the actual return fence as a join; never retarget CPU completion to an earlier device task.
                self.span(scope, boundary["resource"], boundary["end"], boundary["end"], "boundary")
                self.edge(self.endpoint(transfer, "end"), self.endpoint(span, "end"), "signal")

        for task in self.streams.payload:
            source = task.source
            kind = source.record["args"]["Task Type"]
            if kind in {"EVENT_RECORD", "EVENT_WAIT", "PROFILING_ENABLE", "PROFILING_DISABLE"}:
                continue
            if source.service is not None or self.trace.auxiliary(source) or self.trace.outside_window(source):
                continue

            stream = task.stream
            resource = self.resource("stream", stream, "execution")
            if self.device_index.enclosing(Interval(task.begin, task.end, stream, {})) is None:
                raise ValueError(f"native payload has no semantic stream coverage: stream {stream} at {task.begin}")

            self.payload.append(task)
            self.resources[resource]["payload"].append([task.begin, task.end])

    def collective_spans(self) -> None:
        groups = defaultdict(dict)
        for task in self.payload:
            api = self.trace.apis.get(task.source.record["args"]["connection_id"])
            if api is None:
                continue

            queue = self.trace.dequeue_ranges.enclosing(api)
            if queue is None or not queue.record["name"].startswith("Dequeue@Hccl"):
                continue
            scope = self.trace.owner(self.trace.origin(api))

            if scope is None:
                continue
            raw = self.raw["scopes"][scope]
            if not uses_peer_service(raw):
                continue
            if raw.get("asynchronous", False) and raw["communication"]["operation"] != "transfer":
                continue

            groups[scope].setdefault(queue.record["args"]["correlation_id"], []).append(task)

        spans = defaultdict(list)
        for span in self.spans:
            if span["kind"] == "communication" and self.resources[span["resource"]]["kind"] == "stream":
                spans[span["scope"]].append(span)

        self.local_peer_services(spans)

        for scope, calls in groups.items():
            if len(calls) != 1:
                continue
            ((correlation, tasks),) = calls.items()
            stream = tasks[0].stream

            if any(task.stream != stream for task in tasks):
                continue
            begin, end = tasks[0].begin, tasks[-1].end
            limit = self.device_index.enclosing(Interval(begin, end, stream, {}))

            if limit is None:
                continue
            enclosing = self.spans[limit.record["span"]]
            if enclosing["scope"] != scope or enclosing["kind"] != "communication":
                continue

            # Keep caller-stream work such as the gather transpose outside the peer-gated service.
            for span in spans[scope]:
                span["kind"] = "device"
            resource = enclosing["resource"]
            if (enclosing["begin"], enclosing["end"]) == (begin, end):
                enclosing["kind"] = "collective"
            else:
                self.span(scope, resource, begin, end, "collective")

            self.collectives[scope] = (resource, begin, end, self.trace.dequeue[correlation])
            if self.evidence and (signals := self.trace.notifications.observed(tasks, self.trace.clock)):
                self.collective_notifications[str(scope)] = signals

        self.device_index = self.span_index("stream")

    def local_peer_services(self, spans) -> None:
        first_wait = {}
        for waiter, producer in self.waited_work.items():
            if self.raw["scopes"][producer].get("local_exchanges"):
                returned = self.raw["scopes"][waiter]["device_return"]
                first_wait[producer] = min(first_wait.get(producer, returned), returned)

        for scope, raw in enumerate(self.raw["scopes"]):
            if "local_exchanges" not in raw:
                continue
            kind = "collective" if uses_peer_service(raw) else "communication"
            resource = self.spans[self.device[scope]]["resource"]

            for span in spans[scope]:
                span["kind"] = "device"
            for call in raw["local_exchanges"]:
                first, last = (self.trace.device_points[call[key]] for key in ("begin", "end"))
                if first.stream != last.stream or self.resource("stream", first.stream, "execution") != resource:
                    raise ValueError("a local protocol cut must keep its caller stream")

                posted = self.host_anchor(first.host)
                command = None
                if raw.get("asynchronous"):
                    if scope not in first_wait or scope in self.peer_services:
                        raise ValueError(f"staged Work {scope} needs one posted operand and a consuming wait")

                    # Post and wait identify the handle's lifetime. A target service can finish independently of the
                    # caller.
                    peer = self.resource("service", "peer", "peer_transport", serial=False)
                    service = self.span(scope, peer, call["begin"], first_wait[scope], kind)
                    self.peer_services[scope] = service
                elif kind == "collective":

                    # HCCL uses the process group's queue for synchronous peer services, including exchanges with
                    # unequal splits.
                    lane = raw["communication"]["channel"]
                    peer = self.resource("service", lane, "peer_collective")
                    service = self.span(scope, peer, call["begin"], call["end"], kind)
                    command = self.staged_boundary(scope, last)
                    self.edge(self.endpoint(service, "end"), command, "signal")
                else:
                    begin, end = self.streams.task(first.task).end, self.streams.task(last.task).end
                    service = self.span(scope, resource, begin, end, kind)

                self.spans[service]["protocol_operand"] = {key: call[key] for key in ("operation", "bytes", "dtype")}

                ready = self.endpoint(service, "begin")
                if self.spans[service]["resource"] != resource:
                    operand = self.staged_boundary(scope, first)
                    self.edge(operand, ready, "data")
                    ready = operand

                # One synchronous stream command is accepted at its caller cut and retires with the result.
                if command is not None:
                    self.edge(posted, ready, "control")
                    self.edge(posted, command, "submission")
                else:
                    self.edge(posted, ready, "submission")

    def auxiliary_stream_spans(self) -> None:
        pending = None
        communications = []
        for task in self.streams.commands:
            source = task.source
            if isinstance(source, NativeStreamCall):
                resource = self.resource("stream", task.stream, "execution")
                self.span(source.scope, resource, task.begin, task.end, "wait")
                pending = None
                continue
            else:
                if source.service is not None or self.trace.auxiliary(source) or self.trace.outside_window(source):
                    continue
                api = self.trace.apis.get(source.record["args"]["connection_id"])
                owner = None if api is None else self.trace.owner(self.trace.origin(api))

            stream = task.stream
            if self.device_index.enclosing(Interval(task.begin, task.end, stream, {})) is not None:
                pending = None
                continue

            if owner is None:
                pending = None
                continue
            resource = self.resource("stream", stream, "execution")
            key = owner, stream

            if pending is not None and pending[0] == key:
                self.spans[pending[1]]["end"] = task.end
            else:
                communication = "communication" in self.raw["scopes"][owner]
                span = self.span(owner, resource, task.begin, task.end, "communication" if communication else "device")
                pending = key, span

                if communication:
                    communications.append((owner, span))

        # Other streams belong to the submitting scope; their actual event/wait records supply completion dependencies.
        for scope, span in communications:

            # Native tensor collectives publish usable output on their caller stream, even with internal communication
            # streams.
            caller = self.device[scope]
            self.edge(self.endpoint(caller, "begin"), self.endpoint(span, "begin"), "data")
            if not self.raw["scopes"][scope].get("asynchronous", False):
                self.edge(self.endpoint(span, "end"), self.endpoint(caller, "end"), "signal")

    def resolve_work_completions(self) -> None:
        producers = {
            scope
            for scope, raw in enumerate(self.raw["scopes"])
            if raw.get("asynchronous", False) and scope not in self.peer_services
        }

        if not producers:
            return
        for native, recorded in self.trace.synchronizations:
            if recorded is None:
                continue
            owner = self.trace.owner(self.trace.origin(self.trace.native_api(native)))
            producer = self.waited_work.get(owner)

            if producer not in producers:
                continue
            api = self.trace.native_api(recorded)
            if self.trace.owner(self.trace.origin(api)) != producer:
                continue

            (task,) = self.trace.tasks[api.record["args"]["connection_id"]]
            task = self.streams.task(task)
            previous = self.work_completions.setdefault(producer, task)
            if previous is not task:
                raise ValueError(f"native Work {producer} waits on more than one completion event")

        for scope in producers:
            if scope not in self.work_completions:
                raise ValueError(f"native Work {scope} has no observed completion wait")

    def host_spans(self) -> None:
        queries = {query["scope"]: query for query in self.raw["state_queries"]}

        # Coordinates express ordering, not service time; observer calls must not fragment business CPU work.
        for scope, interval in self.trace.scopes.items():
            if self.raw["scopes"][scope]["thread"] != interval.thread:
                raise ValueError(f"scope {scope} differs from its recorded native thread")
            resource = self.resource("thread", interval.thread, "native")
            raw = self.raw["scopes"][scope]
            kind = "wait" if "service" in raw else "boundary" if raw.get("envelope", False) else "host"

            if "communication" in raw and "device_entry" not in raw:
                kind = "collective" if uses_peer_service(raw) else "communication"
                if interval.begin < interval.end:
                    self.resources[resource]["payload"].append([interval.begin, interval.end])

            self.host[scope] = self.span(scope, resource, interval.begin, interval.end, kind)
            if scope in queries:
                read = self.trace.native_interval({**queries[scope], "thread": interval.thread})
                if not interval.contains(read):
                    raise ValueError(f"query {scope} escapes its observed caller interval")

                # The actual read returns before observer bookkeeping; retain its caller envelope without charging it.
                self.callers[scope] = self.host[scope]
                self.spans[self.host[scope]]["kind"] = "boundary"
                self.host[scope] = self.span(scope, resource, read.begin, read.end, "host")

        self.host_index = self.span_index("thread")
        for dequeue in sorted(self.trace.dequeue.values(), key=lambda value: (value.thread, value.begin)):
            enqueue = self.trace.enqueue[dequeue.record["args"]["correlation_id"]]
            if self.trace.excluded_ranges.enclosing(enqueue) is not None:
                continue

            scope = self.trace.owner(enqueue)
            if scope is None or scope in self.trace.closed_scopes:
                continue

            # Handle maintenance remains lifecycle evidence, not an extra submission segment between business tasks.
            apis = self.trace.api_ranges.inside(dequeue)
            if not any(
                self.trace.tasks.get(api.record["args"]["connection_id"])
                or api.record["args"]["connection_id"] in self.streams.stream_calls
                or api.record["args"]["id"] in HOST_WAITS | {"aclrtMemcpy", "aclrtMemcpyAsync"}
                for api in apis
            ):
                continue

            resource = self.resource("thread", dequeue.thread, "native")

            # Each take has its own producer, even when adjacent commands belong to the same scope.
            span = self.span(scope, resource, dequeue.begin, dequeue.end, "host")
            self.edge(self.host_anchor(enqueue), self.endpoint(span, "begin"), "submission")

        for native in self.trace.runtime:
            if native.record["api"] != "aclrtMemcpy" or native.record["bytes"] == 0:
                continue

            # The observation window covers every worker thread, not only its opening caller.
            if native.begin < self.trace.window.begin or self.trace.window.end < native.end:
                continue
            origin = self.trace.origin(self.trace.native_api(native))
            owner = self.trace.owner(origin)

            if (
                owner is not None
                and owner not in self.trace.closed_scopes
                and self.trace.excluded_ranges.enclosing(origin) is None
            ):
                operand = SynchronousCopy.operand(native.record, self.raw["scopes"][owner])
                self.host_transfer(owner, native, native.record["direction"], operand)

        self.host_index = self.span_index("thread")
