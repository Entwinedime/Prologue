"""Connect call ports, submissions and waits to observed native endpoints."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

from prologue.integrations.platforms.ascend.analysis.commands import NativeStreamCall, StreamCommand
from prologue.integrations.platforms.ascend.trace_store import HOST_WAITS, Interval

from .commands import span_references


class PortBindings:
    def __init__(self, graph):
        self.graph = graph

    def resolve_work_completions(self) -> None:
        graph = self.graph
        producers = {scope for scope, raw in enumerate(graph.raw["scopes"])
                     if raw.get("asynchronous", False) and scope not in graph.peer_services}
        if not producers:
            return
        for native, recorded in graph.trace.synchronizations:
            if recorded is None:
                continue
            owner = graph.trace.owner(graph.trace.origin(graph.trace.native_api(native)))
            producer = graph.waited_work.get(owner)
            if producer not in producers:
                continue
            api = graph.trace.native_api(recorded)
            if graph.trace.owner(graph.trace.origin(api)) != producer:
                continue
            task, = graph.trace.tasks[api.record["args"]["connection_id"]]
            task = graph.streams.task(task)
            previous = graph.work_completions.setdefault(producer, task)
            if previous is not task:
                raise ValueError(f"native Work {producer} waits on more than one completion event")

        for scope in producers:
            if scope not in graph.work_completions:
                raise ValueError(f"native Work {scope} has no observed completion wait")

    def call_ports(self) -> None:
        graph = self.graph
        observed = []
        for scope, raw in enumerate(graph.raw["scopes"]):
            if scope == 0:
                continue
            host = graph.host[scope]
            ports = graph.scopes[scope]["ports"]
            for name, direction, side in (("call.enter", "input", "begin"), ("call.return", "output", "end")):
                ports.append({
                    "name": name, "direction": direction, "use": "control", "relation": "control",
                    "at": graph.endpoint(host, side),
                })
            if scope in graph.callers:
                ports.append({
                    "name": "caller.return", "direction": "output", "use": "control", "relation": "control",
                    "at": graph.endpoint(graph.callers[scope], "end"),
                })
            if scope in graph.services:
                ports.append({
                    "name": "service.complete", "direction": "output", "use": "control", "relation": "control",
                    "at": graph.endpoint(graph.services[scope], "end"),
                })
            if scope in graph.device:
                for name, direction, side in (("device.enter", "input", "begin"), ("device.complete", "output", "end")):
                    ports.append({
                        "name": name, "direction": direction, "use": "control", "relation": "control",
                        "at": graph.endpoint(graph.device[scope], side),
                    })
            if "communication" in raw:
                span = graph.device[scope] if "device_entry" in raw else host
                completion = graph.work_end(scope) if raw.get("asynchronous", False) else graph.endpoint(span, "end")
                ports.append(
                    {
                        "name": "communication.complete",
                        "direction": "output",
                        "use": "control",
                        "relation": "communication",
                        "at": completion,
                        "rendezvous": graph.endpoint(span, "begin"),
                    }
                )
            for port in raw["ports"]:
                if port["use"] == "signal":
                    if "boundary" in port:
                        boundary = port["boundary"]
                        span = {"call": graph.host, "device": graph.device, "service": graph.services}[
                            boundary["interval"]
                        ][boundary["scope"]]
                        ports.append({
                            "name": port["name"], "direction": port["direction"], "use": "signal", "relation": "signal",
                            "signal": f"scope/{boundary['scope']}/{boundary['interval']}/{boundary['side']}",
                            "at": graph.endpoint(span, boundary["side"]),
                        })
                        continue
                    if (producer := port.get("producer")) in graph.peer_services:
                        completed = graph.work_end(producer)
                        at = completed
                        if port["direction"] == "input":
                            entered, returned = (
                                graph.trace.device_points[raw[f"device_{side}"]] for side in ("entry", "return")
                            )

                            # The entry marker is submitted before the measured host call. Only completion joins it.
                            graph.staged_boundary(scope, entered)
                            at = graph.staged_boundary(scope, returned)
                            graph.edge(graph.endpoint(host, "begin"), at, "control")
                            graph.edge(completed, at, "signal")
                        ports.append({"name": port["name"], "direction": port["direction"], "use": "signal",
                                      "relation": "signal", "signal": f"work/{producer}", "at": at})
                        continue
                    task = (
                        graph.work_completions[port["producer"]]
                        if "producer" in port
                        else graph.streams.task(graph.trace.device_points[port["point"]].task)
                    )
                    connection = task.source.record["args"]["connection_id"]
                    if port["direction"] == "input":
                        graph.signal_inputs[scope, connection] = port["name"]
                    else:
                        ports.append({
                            "name": port["name"], "direction": "output", "use": "signal", "relation": "signal",
                            "signal": f"event/{connection}", "at": graph.task_anchor(task),
                        })
                        self.submission(task)
                    continue
                side = "end" if port["direction"] == "output" or port.get("at_return", False) else "begin"
                span = graph.device[scope] if port["use"] in {"device", "state"} else host
                tensor = port["tensor"]
                use = port["use"]
                if use in {"host", "device"}:
                    use = "data" if tensor is not None else "control"
                elif use == "state":

                    # Persistent buffer ports describe layout; state_accesses owns the actual read/write intervals.
                    use = "metadata"
                relation = "data" if use == "data" else "control"
                converted = {
                    "name": port["name"], "direction": port["direction"], "use": use,
                    "relation": relation, "at": graph.endpoint(span, side),
                }
                if raw.get("asynchronous", False) and port["use"] == "device" and port["direction"] == "output":
                    converted["at"] = graph.work_end(scope)
                if tensor is not None:
                    converted["tensor"] = {
                        name: tensor[name]
                        for name in ("value", "storage", "version", "dtype", "shape", "writer", "allocation", "buffer")
                    }
                    converted["tensor"].update(
                        (name, tensor[name]) for name in ("shape_rule", "packing") if name in tensor
                    )
                ports.append(converted)
                if tensor is not None and port["use"] not in {"state", "metadata"}:
                    observed.append((getattr(graph.trace.scopes[scope], side), scope, converted))
        self.data_dependencies(observed)

    def data_dependencies(self, observed: list) -> None:
        graph = self.graph

        # Only an already published version can feed another call; an enclosing writer's return is in the future.
        producers = {}
        for _, scope, port in sorted(observed, key=lambda item: item[0]):
            tensor = port["tensor"]
            key = tensor["storage"], tensor["version"]
            if port["direction"] == "output":
                producer, at = producers.setdefault(key, (scope, port["at"]))

                # A parent re-exports the child's ready value, not its own later completion. Siblings keep local ports.
                owner = producer
                while owner is not None and owner != scope:
                    owner = graph.scopes[owner]["parent"]
                if owner == scope:
                    port["at"] = at
            elif key in producers:
                graph.edge(producers[key][1], port["at"], "data")

    def state_accesses(self) -> None:
        graph = self.graph
        for scope, raw in enumerate(graph.raw["scopes"]):
            for access in raw.get("state_accesses", []):
                spans = {"device": graph.device, "call": graph.host, "service": graph.services}[access["interval"]]
                field = {"kv": "kv_accesses", "slot": "slot_accesses"}[access["space"]]
                graph.scopes[scope][field].append({
                    **{name: value for name, value in access.items() if name != "interval"},
                    "start": graph.endpoint(spans[scope], "begin"),
                    "finish": graph.endpoint(spans[scope], "end"),
                })

    def submission(self, task: StreamCommand) -> None:
        graph = self.graph
        api, post = graph.streams.posting(task)
        origin = graph.trace.origin(api)
        source = task.source
        side = (
            "end"
            if isinstance(source, NativeStreamCall) or source.record["args"]["Task Type"] == "EVENT_RECORD"
            else "begin"
        )

        # Queue entry follows any CPU preparation inside the API; its outer entry is not a launch witness.
        launched = graph.host_anchor(Interval(post.begin, post.begin, post.thread, {}))
        if origin is not api:

            # A separate enqueue is visible before the worker submits. A direct call enters the queue at the driver
            # boundary.
            graph.edge(graph.host_anchor(origin), launched, "submission")
        graph.edge(launched, graph.task_anchor(task, side), "submission")

    def submission_tail(self, task: StreamCommand) -> None:
        graph = self.graph
        at = graph.task_anchor(task)
        span = graph.spans[at["span"]]

        # Charge up to native payload completion, excluding the next command's queue residence.
        graph.span(span["scope"], span["resource"], task.end, task.end, "boundary")

    def submissions(self) -> None:
        graph = self.graph
        cuts: dict[int, set[int]] = defaultdict(set)
        for span in graph.spans:
            if graph.resources[span["resource"]]["kind"] == "stream":
                cuts[span["resource"]].update((span["begin"], span["end"]))
        for at in span_references({"scopes": graph.scopes, "dependencies": graph.dependencies}):
            resource = graph.spans[at["span"]]["resource"]
            if resource in cuts:
                cuts[resource].add(at["order"])
        ordered = {resource: sorted(positions) for resource, positions in cuts.items()}

        prior, last = None, None
        for task in graph.payload:
            enclosing = graph.device_index.enclosing(Interval(task.begin, task.end, task.stream, {}))
            owner = enclosing.record["span"]
            span = graph.spans[owner]
            resource = span["resource"]
            region = bisect_right(ordered[resource], task.begin) - 1

            # Closed collectives own their internal progress. Other work can start only at its actual queue post.
            if span["kind"] == "collective":
                release = ("collective", owner)
            else:
                _, post = graph.streams.posting(task)
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

    def synchronization(self) -> None:
        graph = self.graph
        syncs, waits = HOST_WAITS, graph.trace.stream_waits
        graph.streams.submission_frontiers(graph.staged_boundaries)
        for api in graph.trace.apis.values():
            if api.record["args"]["id"] not in syncs | waits:
                continue
            origin = graph.trace.origin(api)
            owner = graph.trace.owner(origin)
            if (
                owner is None
                or owner in graph.trace.closed_scopes
                or graph.trace.excluded_ranges.enclosing(origin) is not None
            ):
                continue
            native = [
                call for call in graph.trace.runtime_ranges.overlapping(api) if call.record["api"] in syncs | waits
            ]
            if not native:
                raise ValueError(f"required synchronization lacks native resource identity: {api.record['args']['id']}")

        # A child call or exposed endpoint starts a new region, even if its source run submits no device work.
        boundaries: dict[int, set[int]] = defaultdict(set)
        for span in graph.spans:
            if graph.resources[span["resource"]]["kind"] == "thread":
                boundaries[span["resource"]].update((span["begin"], span["end"]))
        anchors = [port["at"] for scope in graph.scopes for port in scope["ports"]]
        anchors.extend(edge[side] for edge in graph.dependencies for side in ("source", "target"))
        for at in anchors:
            resource = graph.spans[at["span"]]["resource"]
            if resource in boundaries:
                boundaries[resource].add(at["order"])
        regions = {resource: sorted(points) for resource, points in boundaries.items()}
        previous_wait = None
        wait_names: dict[int, int] = defaultdict(int)
        for native, recorded in graph.trace.synchronizations:
            row = native.record
            stream_wait = row["api"] in waits
            api = graph.trace.native_api(native)
            origin = graph.trace.origin(api)
            owner = graph.trace.owner(origin)
            if (
                owner is None
                or owner in graph.trace.closed_scopes
                or graph.trace.excluded_ranges.enclosing(origin) is not None
            ):
                continue
            if stream_wait:
                waiting = graph.streams.stream_calls[api.record["args"]["connection_id"]]
                completion = graph.task_anchor(waiting)
                self.submission(waiting)
            else:
                completion = graph.host_anchor(native, "end")

            signal = None
            if "Event" in row["api"]:
                previous_wait = None
                if recorded is None:
                    raise ValueError("required event wait has no observed recording")
                call = graph.trace.native_api(recorded)
                tasks = [graph.streams.task(task) for task in graph.trace.tasks[call.record["args"]["connection_id"]]]
                signal = call.record["args"]["connection_id"]
            else:
                stream = None if row["api"] == "aclrtSynchronizeDeviceWithTimeout" else row["stream"]
                if stream is not None and row["stream_status"] != 0:
                    raise ValueError("required stream synchronization has no native stream identity")
                tasks = graph.streams.submitted(native.begin, stream)

                # Keep the first target-frontier wait; repeated fences without new work add only CPU overhead.
                resource = graph.spans[completion["span"]]["resource"]
                frontier = tuple(sorted((task.stream, task.end) for task in tasks))
                caller = graph.host_anchor(origin)
                caller_resource = graph.spans[caller["span"]]["resource"]
                region = (
                    bisect_right(regions[resource], native.begin),
                    bisect_right(regions[caller_resource], origin.begin),
                )
                key = owner, completion["span"], caller["span"], stream, region, frontier
                if key == previous_wait:
                    graph.repeated_host_waits.add((native.thread, native.begin))
                    continue
                previous_wait = key

                # The source edge proves this run; the port preserves the operation for a different target frontier.
                peers = [graph.resource("stream", stream, "execution")] if stream is not None else [
                    identity for identity, resource in enumerate(graph.resources) if resource["kind"] == "stream"
                ]
                for peer in peers:
                    graph.scopes[owner]["ports"].append({
                        "name": f"stream.wait/{wait_names[owner]}", "direction": "input", "use": "control",
                        "relation": "signal", "at": completion,
                        "wait": {"resource": peer, "begin": graph.host_anchor(native)},
                    })
                    wait_names[owner] += 1
            if not stream_wait:
                resource = graph.spans[completion["span"]]["resource"]
                graph.span(owner, resource, native.begin, native.end, "wait")
            if (owner, signal) in graph.signal_inputs:
                graph.scopes[owner]["ports"].append(
                    {
                        "name": graph.signal_inputs.pop((owner, signal)),
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
                if "Event" in row["api"]:
                    self.submission(task)
                graph.edge(graph.task_anchor(task), completion, "signal")

        if graph.signal_inputs:
            raise ValueError(f"declared event waits have no native synchronization evidence: {graph.signal_inputs}")
