"""Project an Ascend recording to the common observation document."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prologue.common.observation.thread_service import ThreadService
from prologue.integrations.platforms.ascend.analysis.services import SynchronousCopy, uses_peer_service

from .device_events import DeviceEvents
from .port_bindings import PortBindings
from .sdk_rows import HOST_WAITS
from .timeline import ObservedTimeline


class AscendObservation:
    def __init__(self, directory: Path, rank: int, raw: dict, trace: Path, *, evidence: bool) -> None:
        self.graph = ObservedTimeline(directory, rank, raw, trace, evidence=evidence)
        self.events = DeviceEvents(self.graph)
        self.ports = PortBindings(self.graph)

    def name_streams(self, points: dict[int, str], waits: dict[int, dict[str, str]]) -> None:
        graph = self.graph
        if not points and not waits:
            return
        trace = graph.trace
        declarations = [(trace.device_points[point].stream, role) for point, role in points.items()]
        observed = set()
        for native, recorded in trace.synchronizations:
            if native.record["api"] not in trace.stream_waits:
                continue
            owner = trace.owner(trace.origin(trace.native_api(native)))
            if owner not in waits:
                continue
            if recorded is None:
                raise ValueError("named stream wait has no native event generation")
            task, = trace.tasks[trace.native_api(recorded).record["args"]["connection_id"]]
            if (task.record["args"]["Task Type"] != "EVENT_RECORD" or recorded.record["stream_status"] != 0
                    or recorded.record["stream"] != task.stream or native.record["stream_status"] != 0):
                raise ValueError("named stream wait changes its recorded producer stream")
            declarations.extend(
                ((task.stream, waits[owner]["source"]), (native.record["stream"], waits[owner]["target"]))
            )
            observed.add(owner)
        if observed != waits.keys():
            raise ValueError("logical stream wait has no observed native wait")

        # The framework's canonical names must identify the same lanes throughout this worker's observation.
        named = {}
        for stream, role in declarations:
            if graph.stream_names.setdefault(stream, role) != role or named.setdefault(role, stream) != stream:
                raise ValueError("native logical streams have conflicting resource bindings")

    def host_spans(self) -> None:
        graph = self.graph
        queries = {query["scope"]: query for query in graph.raw["state_queries"]}

        # Coordinates express ordering, not service time; observer calls must not fragment business CPU work.
        for scope, interval in graph.trace.scopes.items():
            if graph.raw["scopes"][scope]["thread"] != interval.thread:
                raise ValueError(f"scope {scope} differs from its recorded native thread")
            resource = graph.resource("thread", interval.thread, "native")
            raw = graph.raw["scopes"][scope]
            kind = "wait" if "service" in raw else "boundary" if raw.get("envelope", False) else "host"
            if "communication" in raw and "device_entry" not in raw:
                kind = "collective" if uses_peer_service(raw) else "communication"
                if interval.begin < interval.end:
                    graph.resources[resource]["payload"].append([interval.begin, interval.end])
            graph.host[scope] = graph.span(scope, resource, interval.begin, interval.end, kind)
            if scope in queries:
                read = graph.trace.native_interval({**queries[scope], "thread": interval.thread})
                if not interval.contains(read):
                    raise ValueError(f"query {scope} escapes its observed caller interval")

                # The actual read returns before observer bookkeeping; retain its caller envelope without charging it.
                graph.callers[scope] = graph.host[scope]
                graph.spans[graph.host[scope]]["kind"] = "boundary"
                graph.host[scope] = graph.span(scope, resource, read.begin, read.end, "host")
        graph.host_index = graph.span_index("thread")
        for dequeue in sorted(graph.trace.dequeue.values(), key=lambda value: (value.thread, value.begin)):
            enqueue = graph.trace.enqueue[dequeue.record["args"]["correlation_id"]]
            if graph.trace.excluded_ranges.enclosing(enqueue) is not None:
                continue
            scope = graph.trace.owner(enqueue)
            if scope is None or scope in graph.trace.closed_scopes:
                continue

            # Handle maintenance remains lifecycle evidence, not an extra submission segment between business tasks.
            apis = graph.trace.api_ranges.inside(dequeue)
            if not any(graph.trace.tasks.get(api.record["args"]["connection_id"])
                       or api.record["args"]["connection_id"] in graph.streams.stream_calls
                       or api.record["args"]["id"] in HOST_WAITS | {"aclrtMemcpy", "aclrtMemcpyAsync"} for api in apis):
                continue
            resource = graph.resource("thread", dequeue.thread, "native")

            # Each take has its own producer, even when adjacent commands belong to the same scope.
            span = graph.span(scope, resource, dequeue.begin, dequeue.end, "host")
            graph.edge(graph.host_anchor(enqueue), graph.endpoint(span, "begin"), "submission")

        for native in graph.trace.runtime:
            if native.record["api"] != "aclrtMemcpy" or native.record["bytes"] == 0:
                continue

            # The observation window covers every worker thread, not only its opening caller.
            if native.begin < graph.trace.window.begin or graph.trace.window.end < native.end:
                continue
            origin = graph.trace.origin(graph.trace.native_api(native))
            owner = graph.trace.owner(origin)
            if (
                owner is not None
                and owner not in graph.trace.closed_scopes
                and graph.trace.excluded_ranges.enclosing(origin) is None
            ):
                operand = SynchronousCopy.operand(native.record, graph.raw["scopes"][owner])
                graph.host_transfer(owner, native, native.record["direction"], operand)
        graph.host_index = graph.span_index("thread")

    def host_service_positions(self, coordinates):
        graph = self.graph
        trace = graph.trace

        def wall_interval(interval):
            identity = f"thread/{interval.thread}"
            return interval.thread, trace.host_times[identity, interval.begin], trace.host_times[identity, interval.end]

        excluded = [wall_interval(interval) for interval in trace.excluded]
        excluded.extend(
            trace.reserved_host_intervals((interval.record for interval in trace.runtime), graph.repeated_host_waits)
        )
        dequeues = []
        for correlation, dequeue in trace.dequeue.items():
            enqueue = trace.enqueue[correlation]
            destination = excluded if trace.excluded_ranges.enclosing(enqueue) is not None else dequeues
            destination.append(wall_interval(dequeue))

        with graph.activity_path.open() as lines:
            service = ThreadService((json.loads(line) for line in lines), dequeues, excluded)
        positions = defaultdict(list)
        for row in coordinates:
            lane = row["resource"]
            resource = graph.resources[lane]
            if resource["kind"] != "thread":
                continue
            thread = int(resource["identity"].split("/")[1])
            observed = service.activity.get(thread)
            if observed is not None and observed[0] <= row["time_ns"] <= observed[1]:
                positions[lane, thread].append((row["order"], row["time_ns"]))

        # Differences of the prefix give the occupied service at any retained cut on the same thread.
        result = []
        for (lane, thread), cuts in positions.items():
            total = 0
            before = cuts[0][1]
            for order, timestamp in cuts:
                if timestamp < before:
                    raise ValueError("raw host cuts reverse execution order")
                if timestamp > before:
                    total += service.duration(thread, before, timestamp, thread not in graph.control_threads)
                result.append([lane, order, total])
                before = timestamp
        return result

    def document(self) -> dict[str, Any]:
        graph = self.graph
        self.host_spans()
        self.events.service_spans()
        self.events.device_spans()
        self.events.collective_spans()
        self.ports.resolve_work_completions()
        self.ports.call_ports()
        self.ports.state_accesses()
        self.ports.synchronization()

        # A true read observes completion without waiting; false reads add no producer dependency.
        for query in graph.raw["state_queries"]:
            if query["kind"] == "event" and query["result"]:
                task = graph.streams.task(graph.trace.device_points[query["point"]].task)
                graph.edge(graph.task_anchor(task), graph.endpoint(graph.host[query["scope"]], "end"), "observation")

        self.ports.submissions()

        # All producers declare payload; publish one union per resource before projection uses ordered ranges.
        for resource in graph.resources:
            ranges = sorted(resource["payload"])
            resource["payload"] = []
            for begin, end in ranges:
                payload = resource["payload"]
                if payload and begin <= payload[-1][1]:
                    payload[-1][1] = max(payload[-1][1], end)
                elif begin < end:
                    payload.append([begin, end])
        result = {
            "environment": graph.raw["environment"], "resources": graph.resources, "scopes": graph.scopes,
            "spans": graph.spans, "dependencies": graph.dependencies,
        }
        self.events.normalize_device_spans(result)
        if graph.evidence:
            result["measurements"] = graph.streams.measurements(result)
            if graph.repeated_host_waits:
                result["measurements"]["repeated_host_waits"] = sorted(graph.repeated_host_waits)
            if graph.activity_path is not None:
                result["measurements"]["host_service_positions"] = self.host_service_positions(
                    result["measurements"]["coordinates"]
                )
            payloads = graph.trace.collective_payloads({scope: row[-1] for scope, row in graph.collectives.items()})
            collectives = []
            for scope, (resource, begin, end, _) in graph.collectives.items():
                payload = payloads.get(scope)

                # An observed variable exchange can report zero sent elements and still complete a peer service.
                if payload is not None and not payload[2]:
                    rows = graph.scopes[scope]["communication"].get("peer_rows")
                    if rows is None or any(rows["send"]):
                        payload = None
                collectives.append([scope, resource, begin, end, payload])
            result["measurements"]["collectives"] = collectives
            if graph.collective_notifications:
                result["measurements"]["collective_notifications"] = graph.collective_notifications
            result["measurements"]["synchronous_transfers"] = self.events.transfer_times()
            result["native_effects"] = graph.streams.evidence(graph.resource_ids)
        return result
