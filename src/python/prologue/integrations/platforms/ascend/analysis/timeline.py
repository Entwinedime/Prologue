"""Own the recorded graph and its native coordinate indexes."""

from __future__ import annotations

from pathlib import Path

from prologue.integrations.platforms.ascend.analysis.commands import StreamCommand, StreamCommands
from prologue.integrations.platforms.ascend.analysis.trace import AscendTrace, DevicePoint, Interval, ThreadIntervals


class ObservedTimeline:
    def __init__(self, directory: Path, rank: int, raw: dict, trace: Path, *, evidence: bool) -> None:
        self.raw = raw
        self.rank = rank
        self.evidence = evidence
        if raw["environment"]["parallel"]["rank"] != rank:
            raise ValueError("worker recording differs from the requested rank")
        activity = raw["host_observation"].get("activity")
        if activity is not None and activity != f"rank-{rank}-thread-activity.jsonl":
            raise ValueError("CPU activity file differs from the observed rank")
        self.activity_path = None if activity is None else directory / activity
        profile, = (directory / "native" / f"rank-{rank}").glob("*_ascend_pt")
        self.trace = AscendTrace(profile, directory / "runtime", raw, trace, evidence=evidence)
        self.streams = StreamCommands(self.trace)
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
        self.repeated_host_waits = set()
        self.signal_inputs = {}
        self.work_completions = {}
        self.peer_services = {}
        self.staged_boundaries: dict[tuple[int, int], dict[str, int]] = {}
        self.waited_work = {
            scope["id"]: port["producer"] for scope in raw["scopes"] for port in scope["ports"]
            if port["use"] == "signal" and port["direction"] == "input" and "producer" in port
        }
        self.scopes = []
        for identity, scope in enumerate(raw["scopes"]):
            if scope["id"] != identity or not scope["complete"]:
                raise ValueError(f"native scope {identity} is not a completed, ordered call")
            info = {name: scope[name] for name in (
                "parent", "role", "site", "implementation", "parameters", "constants", "arguments", "work", "bindings",
            )}
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
        if set(self.trace.scopes) != set(range(1, len(self.scopes))):
            raise ValueError("native scope metadata and recorded CPU markers differ")

    def resource(self, kind: str, identity: int | str, role: str, *, serial: bool = True) -> int:
        key = kind, identity
        if key not in self.resource_ids:
            if kind == "thread":
                role = "control" if identity in self.control_threads else "submission"
            elif kind == "stream":
                role = self.stream_names.get(identity, role)
            self.resource_ids[key] = len(self.resources)
            self.resources.append({
                "kind": kind, "rank": self.rank, "role": role, "identity": f"{kind}/{identity}",
                "serial": serial, "observed": True, "payload": [],
            })
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
