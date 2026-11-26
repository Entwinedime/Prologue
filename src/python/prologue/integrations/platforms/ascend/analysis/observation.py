"""Project the recorded rank into work, ports and qualified causal dependencies."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prologue.common.observation import ThreadService
from prologue.integrations.platforms.ascend.analysis.trace import (
    AscendTrace,
)

from .boundaries import trim_device_boundaries
from .call_ports import ObservedPorts
from .queue_dependencies import QueueDependencies
from .spans import ObservedSpans


class AscendObservation:
    """Compose device measurements and semantic capture into facts for the C++ graph projection."""

    def __init__(self, directory: Path, rank: int, raw: dict, trace: Path, *, evidence: bool) -> None:
        self.evidence = evidence
        if raw["environment"]["parallel"]["rank"] != rank:
            raise ValueError("worker recording differs from the requested rank")

        activity = raw["host_observation"].get("activity")
        if activity is not None and activity != f"rank-{rank}-thread-activity.jsonl":
            raise ValueError("CPU activity file differs from the observed rank")
        self.activity_path = None if activity is None else directory / activity
        (profile,) = (directory / "native" / f"rank-{rank}").glob("*_ascend_pt")
        native_trace = AscendTrace(profile, directory / "runtime", raw, trace, evidence=evidence)
        self.graph = ObservedSpans(rank, raw, native_trace, evidence=evidence)
        self.queues = QueueDependencies(self.graph)

    def name_streams(self, points: dict[int, str], waits: dict[int, dict[str, str]]) -> None:
        if not points and not waits:
            return
        trace = self.graph.trace
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

            (task,) = trace.tasks[trace.native_api(recorded).record["args"]["connection_id"]]
            if (
                task.record["args"]["Task Type"] != "EVENT_RECORD"
                or recorded.record["stream_status"] != 0
                or recorded.record["stream"] != task.stream
                or native.record["stream_status"] != 0
            ):
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
            if self.graph.stream_names.setdefault(stream, role) != role or named.setdefault(role, stream) != stream:
                raise ValueError("native logical streams have conflicting resource bindings")

    def host_service_positions(self, coordinates):
        trace = self.graph.trace

        excluded = [trace.host_interval(interval) for interval in trace.excluded]
        excluded.extend(trace.reserved_host_intervals(self.queues.repeated_host_waits))
        dequeues = []
        for interval, observer_owned in trace.dequeued_intervals():
            (excluded if observer_owned else dequeues).append(interval)

        with self.activity_path.open() as lines:
            service = ThreadService((json.loads(line) for line in lines), dequeues, excluded)
        positions = defaultdict(list)
        for row in coordinates:
            lane = row["resource"]
            resource = self.graph.resources[lane]
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
                    total += service.duration(thread, before, timestamp, thread not in self.graph.control_threads)

                result.append([lane, order, total])
                before = timestamp

        return result

    def document(self) -> dict[str, Any]:
        self.graph.host_spans()
        self.graph.service_spans()
        self.graph.device_spans()
        self.graph.collective_spans()
        self.graph.resolve_work_completions()
        ports = ObservedPorts(self.graph, self.queues)
        ports.call_ports()
        ports.state_accesses()
        self.queues.synchronization(ports.signal_inputs)

        # A true read observes completion without waiting; false reads add no producer dependency.
        for query in self.graph.raw["state_queries"]:
            if query["kind"] == "event" and query["result"]:
                task = self.graph.streams.task(self.graph.trace.device_points[query["point"]].task)
                self.graph.edge(
                    self.graph.task_anchor(task),
                    self.graph.endpoint(self.graph.host[query["scope"]], "end"),
                    "observation",
                )

        self.queues.submissions()

        # All producers declare payload; publish one union per resource before projection uses ordered ranges.
        for resource in self.graph.resources:
            ranges = sorted(resource["payload"])
            resource["payload"] = []
            for begin, end in ranges:
                payload = resource["payload"]
                if payload and begin <= payload[-1][1]:
                    payload[-1][1] = max(payload[-1][1], end)
                elif begin < end:
                    payload.append([begin, end])

        result = {
            "environment": self.graph.raw["environment"],
            "resources": self.graph.resources,
            "scopes": self.graph.scopes,
            "spans": self.graph.spans,
            "dependencies": self.graph.dependencies,
        }
        trim_device_boundaries(result, self.graph.device)

        if self.evidence:
            result["measurements"] = self.graph.streams.measurements(result)
            if self.queues.repeated_host_waits:
                result["measurements"]["repeated_host_waits"] = sorted(self.queues.repeated_host_waits)

            if self.activity_path is not None:
                result["measurements"]["host_service_positions"] = self.host_service_positions(
                    result["measurements"]["coordinates"]
                )

            payloads = self.graph.trace.collective_payloads(
                {scope: row[-1] for scope, row in self.graph.collectives.items()}
            )
            collectives = []

            for scope, (resource, begin, end, _) in self.graph.collectives.items():
                payload = payloads.get(scope)

                # An observed variable exchange can report zero sent elements and still complete a peer service.
                if payload is not None and not payload[2]:
                    rows = self.graph.scopes[scope]["communication"].get("peer_rows")
                    if rows is None or any(rows["send"]):
                        payload = None

                collectives.append([scope, resource, begin, end, payload])

            result["measurements"]["collectives"] = collectives
            if self.graph.collective_notifications:
                result["measurements"]["collective_notifications"] = self.graph.collective_notifications
            result["measurements"]["synchronous_transfers"] = self.graph.transfer_times()
            result["native_effects"] = self.graph.streams.evidence(self.graph.resource_ids)

        return result
