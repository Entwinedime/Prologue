"""Index native marker, queue and connection evidence without building an operator graph."""

from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prologue.integrations.platforms.ascend.analysis.notifications import HcclNotifications
from prologue.integrations.platforms.ascend.trace_store import (
    HOST_WAITS,
    STREAM_WAITS,
    AscendClock,
    HostCounter,
    Interval,
    ThreadIntervals,
    event_records,
    native_interval,
    queue_marker,
    runtime_rows,
    sdk_intervals,
    sdk_rows,
    task_flips,
)

from .queues import NativeQueues, ServiceExecutions


@dataclass
class DeviceTask:
    begin: int
    end: int
    record: dict[str, Any]
    service: int | None = None

    @property
    def stream(self) -> int:
        return int(self.record["args"]["Physic Stream Id"])


@dataclass
class DevicePoint:
    host: Interval
    api: Interval
    task: DeviceTask
    stream: int

class AscendTrace:
    stream_waits = STREAM_WAITS

    @classmethod
    def reserved_host_intervals(cls, calls, repeated_host_waits):
        for call in calls:
            identity = call["thread"], call["begin_counter"]
            if identity in repeated_host_waits:
                continue
            if call["api"] in HOST_WAITS or (call["api"] == "aclrtMemcpy" and call["bytes"]):
                yield call["thread"], call["begin_raw_ns"], call["end_raw_ns"]

    def __init__(self, profile: Path, runtime: Path, raw: dict, trace: Path, *, evidence=False) -> None:
        native_points = {index for index, point in enumerate(raw["points"]) if point["side"] == "signal"}
        services = [scope for scope in raw["scopes"] if "service" in scope]
        self.closed_scopes = set()
        for scope in raw["scopes"]:
            if "service" in scope or scope["parent"] in self.closed_scopes:
                self.closed_scopes.add(scope["id"])
        host_observation = raw["host_observation"]

        # Host counters and stream ordinals establish order; both native and SDK times reach the common RAW clock.
        counter = HostCounter(profile)
        self.cann, = profile.glob("PROF_*")
        device, = self.cann.glob("device_*")
        self.clock = AscendClock(device, raw["device_clock"])
        rows = sdk_rows(trace)
        self.notifications = HcclNotifications(rows) if evidence else None
        flips = task_flips(profile)
        intervals = list(sdk_intervals(rows, counter, self.clock))
        self.host_times = {
            (f"thread/{interval.thread}", position): counter.timestamp(position)
            for interval in intervals if "Task Type" not in interval.record.get("args", {})
            for position in (interval.begin, interval.end)
        }

        # Direct RAW observations take precedence when both recordings name the same counter coordinate.
        native_intervals = [self.native_interval(row) for row in host_observation["intervals"]]
        self.process = host_observation["process"]
        self.scopes = {}
        self.points = {}
        windows = []
        self.excluded = []
        enqueues, dequeues = [], []
        self.apis: dict[int, Interval] = {}
        self.tasks: dict[int, list[DeviceTask]] = defaultdict(list)
        self.device_tasks = []
        streams = defaultdict(list)
        for interval in [*native_intervals, *intervals]:
            row = interval.record
            name = row["name"]
            if name == "prologue.window":
                windows.append(interval)
            elif name.startswith("prologue.scope/"):
                identity = int(name.split("/")[1])
                if identity in self.scopes:
                    raise ValueError(f"duplicate semantic scope marker: {identity}")
                self.scopes[identity] = interval
            elif name.startswith("prologue.point/"):
                identity = int(name.split("/")[1])
                if identity in self.points:
                    raise ValueError(f"duplicate device point marker: {identity}")
                self.points[identity] = interval

                # Native Event.record exports a completion boundary; inserted observer fences have no business owner.
                if identity not in native_points:
                    self.excluded.append(interval)
            elif name in {"prologue.observer", "prologue.auxiliary"}:
                self.excluded.append(interval)
            elif row.get("cat") == "enqueue":
                enqueues.append(interval)
            elif row.get("cat") == "dequeue":
                dequeues.append(interval)
            elif "Thread Id" in row.get("args", {}) and "connection_id" in row["args"]:
                connection = row["args"]["connection_id"]
                if connection in self.apis:
                    raise ValueError(f"duplicate native API connection: {connection}")
                self.apis[connection] = interval
            elif "Task Type" in row.get("args", {}):
                args = row["args"]
                if not queue_marker(args, flips):
                    streams[int(args["Physic Stream Id"])].append(interval)

        # A zero-time wait still has distinct entry and completion. Wall-clock coincidence must not change graph
        # topology.
        for tasks in streams.values():
            tasks.sort(
                key=lambda value: (
                    value.begin,
                    value.record["args"]["Batch Id"],
                    value.record["args"]["Task Id"],
                    value.record["args"]["Subtask Id"],
                )
            )
            for order, interval in enumerate(tasks):
                task = DeviceTask(2 * order, 2 * order + 1, interval.record)
                self.tasks[task.record["args"]["connection_id"]].append(task)
                self.device_tasks.append(task)
        if len(windows) != 1:
            raise ValueError("native recording requires one closed observation window")
        self.window, = windows
        for scope in services:
            identity = scope["id"]
            if identity in self.scopes:
                raise ValueError(f"service has two competing host observations: {identity}")
            interval = scope["host_interval"]
            self.scopes[identity] = self.native_interval(
                {**interval, "thread": scope["thread"], "name": f"prologue.scope/{identity}"}
            )
        self.scope_ranges = ThreadIntervals(list(self.scopes.values()))
        self.excluded_ranges = ThreadIntervals(self.excluded)
        self.enqueue_ranges = ThreadIntervals(enqueues)
        self.dequeue_ranges = ThreadIntervals(dequeues)
        self.enqueue = self.queue_index(enqueues)
        self.dequeue = self.queue_index(dequeues)
        self.api_ranges = ThreadIntervals(list(self.apis.values()))

        self.runtime = []
        for row in runtime_rows(runtime, self.process):
            if row["result"] != 0:
                raise RuntimeError(f"native runtime call failed: {row['api']} ({row['result']})")
            self.runtime.append(self.native_interval(row))
        self.queues = NativeQueues(self.runtime, self.apis, rows, flips)
        self.services = ServiceExecutions(services, self.queues)
        for native in self.runtime:
            row = native.record
            if row["api"].startswith("aclrt") and row["stream_status"] == 0 and row["stream"] in self.services.streams:
                raise ValueError(
                    "a private transport queue has external runtime operations; explicit ports are required"
                )
        self.services.assign(self.device_tasks)

        self.runtime_ranges = ThreadIntervals(self.runtime)
        self.synchronizations = list(event_records(self.runtime, HOST_WAITS | self.stream_waits))
        self.device_points = {identity: self.device_point(identity) for identity in self.points}
        self.observer_connections = {
            point.api.record["args"]["connection_id"]
            for identity, point in self.device_points.items()
            if identity not in native_points
        }

    def native_interval(self, record: dict) -> Interval:
        thread = record["thread"]
        for side in ("begin", "end"):
            for resource in (f"thread/{thread}", f"service/{thread}"):
                self.host_times[resource, record[f"{side}_counter"]] = record[f"{side}_raw_ns"]
        return native_interval(record)

    def device_interval(self, record: dict) -> Interval:
        return Interval.read(record, self.clock)

    def collective_payloads(self, queues: dict[int, Interval]) -> dict[int, list[int]]:
        if not queues:
            return {}
        calls = ThreadIntervals([Interval(queue.begin, queue.end, queue.thread, {"scope": scope})
                                 for scope, queue in queues.items()])
        records = defaultdict(list)

        # Match the SDK compact HCCL op record to its own worker-thread dequeue interval.
        layout = struct.Struct("=HHIIIQBBQQQ14x")
        for family in ("aging", "unaging"):
            paths = (self.cann / "host/data").glob(f"{family}.compact.hccl_op_info.slice_*")
            slices = sorted((path for path in paths if path.name.rsplit("_", 1)[-1].isdigit()),
                            key=lambda path: int(path.name.rsplit("_", 1)[1]))
            data = b"".join(path.read_bytes() for path in slices)
            for magic, _, _, thread, _, stamp, _, dtype, algorithm, count, _ in layout.iter_unpack(
                data[len(data) % layout.size :]
            ):
                if magic != 0x5A5A:
                    raise ValueError("invalid SDK HCCL op record")
                owner = calls.enclosing(Interval(stamp, stamp, thread, {}))
                if owner is not None:
                    records[owner.record["scope"]].append([dtype, algorithm, count])
        return {scope: rows[0] for scope, rows in records.items() if len(rows) == 1}


    @staticmethod
    def queue_index(intervals: list[Interval]) -> dict[int, Interval]:
        result = {}
        for interval in intervals:
            correlation = interval.record["args"]["correlation_id"]
            if correlation in result:
                raise ValueError(f"duplicate native queue correlation: {correlation}")
            result[correlation] = interval
        return result

    def origin(self, api: Interval) -> Interval:
        dequeue = self.dequeue_ranges.enclosing(api)
        if dequeue is None:
            return api
        return self.enqueue[dequeue.record["args"]["correlation_id"]]

    def owner(self, interval: Interval) -> int | None:
        scope = self.scope_ranges.enclosing(interval)
        return None if scope is None else int(scope.record["name"].split("/")[1])

    def device_point(self, identity: int) -> DevicePoint:
        marker = self.points[identity]
        queues = self.enqueue_ranges.inside(marker)
        if len(queues) != 1:
            raise ValueError(f"device point {identity} does not have one native enqueue")
        dequeue = self.dequeue[queues[0].record["args"]["correlation_id"]]
        calls = [api for api in self.api_ranges.inside(dequeue) if api.record["args"]["id"] == "aclrtRecordEvent"]
        if len(calls) != 1:
            raise ValueError(f"device point {identity} does not identify one record API")
        api = calls[0]
        native = self.runtime_call(api)
        if native is None:
            raise ValueError(f"device point {identity} has no recorded API instance")
        tasks = self.tasks[api.record["args"]["connection_id"]]
        if len(tasks) != 1 or tasks[0].record["args"]["Task Type"] != "EVENT_RECORD":
            raise ValueError(f"device point {identity} does not identify one native event task")
        stream = int(tasks[0].record["args"]["Physic Stream Id"])
        if native.record["stream_status"] != 0 or native.record["stream"] != stream:
            raise ValueError(f"device point {identity} disagrees with its native stream")
        return DevicePoint(marker, api, tasks[0], stream)

    def runtime_call(self, api: Interval) -> Interval | None:

        # Both recordings now use the same counter; the interposer encloses the actual SDK API.
        calls = [native for native in self.runtime_ranges.overlapping(api)
                 if native.contains(api) and native.record["api"] == api.record["args"]["id"]]
        if len(calls) > 1:
            raise ValueError(f"ambiguous native API instance: {api.record['args']['id']}")
        return calls[0] if calls else None

    def native_api(self, native: Interval) -> Interval:
        calls = [api for api in self.api_ranges.inside(native) if api.record["args"]["id"] == native.record["api"]]
        if len(calls) != 1:
            raise ValueError(f"native call does not identify one profiler API: {native.record['api']}")
        return calls[0]

    def auxiliary(self, task: DeviceTask) -> bool:
        api = self.apis.get(task.record["args"]["connection_id"])
        return api is not None and self.excluded_ranges.enclosing(self.origin(api)) is not None

    def outside_window(self, task: DeviceTask) -> bool:
        api = self.apis.get(task.record["args"]["connection_id"])
        if api is None:
            return False
        origin = self.origin(api)
        return origin.end <= self.window.begin or self.window.end <= origin.begin
