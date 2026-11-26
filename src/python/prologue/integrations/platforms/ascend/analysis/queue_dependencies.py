"""Lower physical queue posts and signal waits to graph dependencies."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from prologue.integrations.platforms.ascend.analysis.commands import (
    NativeStreamCall,
    StreamCommand,
    span_references,
)
from prologue.integrations.platforms.ascend.trace_store import HOST_WAITS, Interval

from .boundaries import resource_positions


class QueueDependencies:
    """Resolve synchronization edges from recorded command positions and event generations."""

    def __init__(self, graph):
        self.graph = graph
        self.repeated_host_waits = set()

    def submission(self, task: StreamCommand) -> None:
        api, post = self.graph.streams.posting(task)
        origin = self.graph.trace.origin(api)
        source = task.source
        side = (
            "end"
            if isinstance(source, NativeStreamCall) or source.record["args"]["Task Type"] == "EVENT_RECORD"
            else "begin"
        )

        # Queue entry follows any CPU preparation inside the API; its outer entry is not a launch witness.
        launched = self.graph.host_anchor(Interval(post.begin, post.begin, post.thread, {}))
        if origin is not api:

            # A separate enqueue is visible before the worker submits. A direct call enters the queue at the driver
            # boundary.
            self.graph.edge(self.graph.host_anchor(origin), launched, "submission")

        self.graph.edge(launched, self.graph.task_anchor(task, side), "submission")

    def submission_tail(self, task: StreamCommand) -> None:
        at = self.graph.task_anchor(task)
        span = self.graph.spans[at["span"]]

        # Charge up to native payload completion, excluding the next command's queue residence.
        self.graph.span(span["scope"], span["resource"], task.end, task.end, "boundary")

    def submissions(self) -> None:
        graph = self.graph
        anchors = span_references({"scopes": graph.scopes, "dependencies": graph.dependencies})
        ordered = resource_positions(graph.resources, graph.spans, anchors, "stream")

        prior, last = None, None
        for task in self.graph.payload:
            enclosing = self.graph.device_index.enclosing(Interval(task.begin, task.end, task.stream, {}))
            owner = enclosing.record["span"]
            span = self.graph.spans[owner]
            resource = span["resource"]
            region = bisect_right(ordered[resource], task.begin) - 1

            # Closed collectives own their internal progress. Other work can start only at its actual queue post.
            if span["kind"] == "collective":
                release = ("collective", owner)
            else:
                _, post = self.graph.streams.posting(task)
                release = ("driver", post.thread, post.begin)

            key = resource, region, release
            if key != prior:
                if last is not None:
                    self.submission_tail(last)
                self.submission(task)
                prior = key

            last = task

        if last is not None:
            self.submission_tail(last)

    def synchronization(self, signal_inputs) -> None:
        syncs, waits = HOST_WAITS, self.graph.trace.stream_waits
        self.graph.streams.submission_frontiers(self.graph.staged_boundaries)
        self.validate_wait_sources(syncs | waits)

        # A child call or exposed endpoint starts a new region, even if its source run submits no device work.
        graph = self.graph
        anchors = [port["at"] for scope in graph.scopes for port in scope["ports"]]
        anchors.extend(edge[side] for edge in graph.dependencies for side in ("source", "target"))
        regions = resource_positions(graph.resources, graph.spans, anchors, "thread")

        previous_wait = None
        wait_names: dict[int, int] = defaultdict(int)
        for native, recorded in self.graph.trace.synchronizations:
            row = native.record
            stream_wait = row["api"] in waits
            api = self.graph.trace.native_api(native)
            origin = self.graph.trace.origin(api)
            owner = self.graph.trace.owner(origin)

            if (
                owner is None
                or owner in self.graph.trace.closed_scopes
                or self.graph.trace.excluded_ranges.enclosing(origin) is not None
            ):
                continue

            if stream_wait:
                waiting = self.graph.streams.stream_calls[api.record["args"]["connection_id"]]
                completion = self.graph.task_anchor(waiting)
                self.submission(waiting)
            else:
                completion = self.graph.host_anchor(native, "end")

            signal = None
            if "Event" in row["api"]:
                previous_wait = None
                if recorded is None:
                    raise ValueError("required event wait has no observed recording")

                call = self.graph.trace.native_api(recorded)
                tasks = [
                    self.graph.streams.task(task)
                    for task in self.graph.trace.tasks[call.record["args"]["connection_id"]]
                ]
                signal = call.record["args"]["connection_id"]
            else:
                stream = None if row["api"] == "aclrtSynchronizeDeviceWithTimeout" else row["stream"]
                if stream is not None and row["stream_status"] != 0:
                    raise ValueError("required stream synchronization has no native stream identity")

                tasks = self.graph.streams.submitted(native.begin, stream)

                # Keep the first target-frontier wait; repeated fences without new work add only CPU overhead.
                resource = self.graph.spans[completion["span"]]["resource"]
                frontier = tuple(sorted((task.stream, task.end) for task in tasks))
                caller = self.graph.host_anchor(origin)
                caller_resource = self.graph.spans[caller["span"]]["resource"]
                region = (
                    bisect_right(regions[resource], native.begin),
                    bisect_right(regions[caller_resource], origin.begin),
                )
                key = owner, completion["span"], caller["span"], stream, region, frontier

                if key == previous_wait:
                    self.repeated_host_waits.add((native.thread, native.begin))
                    continue
                previous_wait = key

                # The source edge proves this run; the port preserves the operation for a different target frontier.
                peers = (
                    [self.graph.resource("stream", stream, "execution")]
                    if stream is not None
                    else [
                        identity
                        for identity, resource in enumerate(self.graph.resources)
                        if resource["kind"] == "stream"
                    ]
                )

                for peer in peers:
                    self.graph.scopes[owner]["ports"].append(
                        {
                            "name": f"stream.wait/{wait_names[owner]}",
                            "direction": "input",
                            "use": "control",
                            "relation": "signal",
                            "at": completion,
                            "wait": {"resource": peer, "begin": self.graph.host_anchor(native)},
                        }
                    )
                    wait_names[owner] += 1

            if not stream_wait:
                resource = self.graph.spans[completion["span"]]["resource"]
                self.graph.span(owner, resource, native.begin, native.end, "wait")

            self.completion_edges(
                owner, signal, completion, tasks, event_wait="Event" in row["api"], signal_inputs=signal_inputs
            )

        if signal_inputs:
            raise ValueError(f"declared event waits have no native synchronization evidence: {signal_inputs}")

    def validate_wait_sources(self, api_names) -> None:
        for api in self.graph.trace.apis.values():
            if api.record["args"]["id"] not in api_names:
                continue
            origin = self.graph.trace.origin(api)
            owner = self.graph.trace.owner(origin)

            if (
                owner is None
                or owner in self.graph.trace.closed_scopes
                or self.graph.trace.excluded_ranges.enclosing(origin) is not None
            ):
                continue

            native = [
                call for call in self.graph.trace.runtime_ranges.overlapping(api) if call.record["api"] in api_names
            ]
            if not native:
                raise ValueError(f"required synchronization lacks native resource identity: {api.record['args']['id']}")

    def completion_edges(self, owner, signal, completion, tasks, *, event_wait, signal_inputs) -> None:
        if (owner, signal) in signal_inputs:
            self.graph.scopes[owner]["ports"].append(
                {
                    "name": signal_inputs.pop((owner, signal)),
                    "direction": "input",
                    "use": "signal",
                    "relation": "signal",
                    "signal": f"event/{signal}",
                    "at": completion,
                }
            )

        tails = {}
        for task in tasks:
            stream = task.stream
            if stream not in tails or task.end > tails[stream].end:
                tails[stream] = task

        for task in tails.values():

            # Payload submission belongs to its coarse interval, not to the fence's last internal kernel.
            if event_wait:
                self.submission(task)
            self.graph.edge(self.graph.task_anchor(task), completion, "signal")
