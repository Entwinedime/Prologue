"""Index native calls and associate device work with recorded service executions."""

from __future__ import annotations

import struct
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import chain
from math import prod
from pathlib import Path
from typing import Any

from prologue.integrations.platforms.ascend.trace_store import (
    HOST_WAITS,
    STREAM_WAITS,
    Interval,
    QueueSlots,
    RawTraceIndex,
    ThreadIntervals,
    device_streams,
    event_records,
    sdk_rows,
    task_flips,
)


class HcclNotifications:
    def __init__(self, rows):
        self.tasks = {}
        for row in rows:
            if row.get("ph") != "X" or row.get("name") not in {"Notify_Record", "Notify_Wait"}:
                continue
            args = row["args"]
            key = args["stream id"], args["task id"], args["context id"], Decimal(str(row["ts"]))
            side = "record" if row["name"] == "Notify_Record" else "wait"
            endpoint = side, args["src rank"], args["dst rank"], args["notify_id"]

            # An SDK annotation and a device task must identify the same physical subtask, including timer epoch.
            self.tasks[key] = None if key in self.tasks else endpoint

    def observed(self, commands, clock):
        signals = []
        for command in commands:
            row = command.source.record
            args = row["args"]
            key = args["Physic Stream Id"], args["Task Id"], args["Subtask Id"], Decimal(str(row["ts"]))

            if key not in self.tasks:
                continue
            endpoint = self.tasks[key]
            if endpoint is None:
                return []

            # A record can signal after it starts. A wait can finish only after the signal has arrived.
            start, finish = clock.interval(row)
            signals.append([*endpoint, start if endpoint[0] == "record" else finish])

        return signals


def uses_peer_service(scope):
    exchange = scope.get("communication")

    # Variable row exchanges keep their transport service as the active peers change.
    return exchange is not None and ("peer_rows" in scope or exchange["required"] == exchange["members"])


class SynchronousCopy:
    directions = {"host_to_device": 1, "device_to_host": 2}  # aclrtMemcpyKind

    @classmethod
    def matches(cls, native, address, size, direction):
        return (native["direction"] == cls.directions[direction] and native.get("source") == address
                and native["bytes"] == size)

    @classmethod
    def operand(cls, native, scope):
        if native["direction"] not in cls.directions.values() or native["result"] != 0:
            return None
        inputs = ((port["name"], port["tensor"]) for port in scope["ports"]
                  if port["direction"] == "input" and port["tensor"] is not None)
        operands = []

        for name, tensor in chain(inputs, scope.get("local_copies", {}).items()):
            source = tensor.get("copy_source")
            if source and cls.matches(
                native, source["address"], source["element_bytes"] * prod(tensor["shape"]), source["direction"]
            ):
                operands.append({
                    "operand": name, "tensor": {"dtype": tensor["dtype"], "shape": tensor["shape_rule"]},
                    "element_bytes": source["element_bytes"],
                    "caller": source["caller"], "implementation": native["api"], "direction": source["direction"],
                })

        # A coincident size, a slice or ambiguous aliases cannot bind the service to a target port.
        return operands[0] if len(operands) == 1 else None


@dataclass
class ServiceExecution:
    scope: int
    thread: int
    begin: int
    end: int


class ServiceExecutions:
    def __init__(self, scopes: list[dict], queues: QueueSlots) -> None:
        owners = {scope["id"]: scope for scope in scopes}
        self.executions = [
            ServiceExecution(scope["id"], call["thread"], call["begin_counter"], call["end_counter"])
            for scope in scopes for call in scope["service"]["executions"]
        ]
        by_owner = defaultdict(list)

        for index, call in enumerate(self.executions):
            interval = owners[call.scope]["host_interval"]
            if not interval["begin_counter"] <= call.begin <= call.end <= interval["end_counter"]:
                raise ValueError("transport execution escapes its synchronous service boundary")

            by_owner[call.scope, call.thread].append((index, call))

        self.submissions = {}
        queue_owners = {}
        submitted = set()
        for stream, native in queues.submissions:
            row = native.record
            if row["service"] < 0:
                continue

            matches = [
                (index, call) for index, call in by_owner[row["service"], native.thread]
                if call.begin <= native.begin and native.end <= call.end
            ]

            if len(matches) != 1:
                raise ValueError("native queue submission lacks one explicit service execution")
            (index, call), = matches

            # Shared queues must remain ordered by the caller thread retained in the coarse graph.
            owner = owners[call.scope]["thread"]
            if queue_owners.setdefault(stream, owner) != owner:
                raise ValueError("transport queue crosses service callers; its ordering requires an explicit interface")
            submitted.add(index)
            self.submissions.setdefault(stream, []).append((native, index))

        self.streams = set(self.submissions)
        if len(submitted) != len(self.executions):
            raise ValueError("transport execution has no native queue submission evidence")
        if any(stream in self.streams and native.record["service"] < 0 for stream, native in queues.submissions):
            raise ValueError("a transport queue also carries work outside its declared services")

    def assign(self, tasks: list) -> None:
        grouped = {stream: [] for stream in self.streams}
        for task in tasks:
            if task.stream in grouped:
                grouped[task.stream].append(task)

        for stream, submissions in self.submissions.items():
            if any(left[0].end > right[0].begin for left, right in zip(submissions, submissions[1:], strict=False)):
                raise ValueError("concurrent private-queue submissions have no established order")

            # Driver submission sizes define correspondence, independently of CPU/device clock calibration.
            payload = sorted(grouped[stream], key=lambda task: task.begin)
            expected = sum(native.record["tasks"] for native, _ in submissions)
            if len(payload) != expected:
                raise ValueError(f"private queue {stream} submission/task coverage differs: {expected}/{len(payload)}")

            position = 0
            for native, execution in submissions:
                count = native.record["tasks"]
                for task in payload[position:position + count]:
                    task.service = self.executions[execution].scope

                position += count


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


class AscendTrace(RawTraceIndex):
    stream_waits = STREAM_WAITS

    def __init__(self, profile: Path, runtime: Path, raw: dict, trace: Path, *, evidence=False) -> None:
        rows = sdk_rows(trace)
        flips = task_flips(profile)
        super().__init__(profile, runtime, raw, trace.parent, rows, flips)
        self.notifications = HcclNotifications(rows) if evidence else None
        self.tasks: dict[int, list[DeviceTask]] = defaultdict(list)
        self.device_tasks = []

        # The ordinal separates native entry and completion even when both share a timestamp.
        for tasks in device_streams(rows, self.clock, flips).values():
            for order, row in enumerate(tasks):
                task = DeviceTask(2 * order, 2 * order + 1, row)
                self.tasks[task.record["args"]["connection_id"]].append(task)
                self.device_tasks.append(task)

        services = [scope for scope in raw["scopes"] if "service" in scope]
        self.services = ServiceExecutions(services, self.queues)
        for native in self.runtime:
            row = native.record
            if row["api"].startswith("aclrt") and row["stream_status"] == 0 and row["stream"] in self.services.streams:
                raise ValueError(
                    "a private transport queue has external runtime operations; explicit ports are required"
                )

        self.services.assign(self.device_tasks)

        self.synchronizations = list(event_records(self.runtime, HOST_WAITS | self.stream_waits))
        self.device_points = {identity: self.device_point(identity) for identity in self.points}
        self.observer_connections = {
            point.api.record["args"]["connection_id"]
            for identity, point in self.device_points.items()
            if identity not in self.signals
        }

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

    def owner(self, interval: Interval) -> int | None:
        scope = self.scope_ranges.enclosing(interval)
        return None if scope is None else int(scope.record["name"].split("/")[1])

    def device_point(self, identity: int) -> DevicePoint:
        marker = self.points[identity]
        api = self.point_api(identity)
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

    def auxiliary(self, task: DeviceTask) -> bool:
        api = self.apis.get(task.record["args"]["connection_id"])
        return api is not None and self.excluded_ranges.enclosing(self.origin(api)) is not None

    def outside_window(self, task: DeviceTask) -> bool:
        api = self.apis.get(task.record["args"]["connection_id"])
        if api is None:
            return False
        origin = self.origin(api)
        return origin.end <= self.window.begin or self.window.end <= origin.begin
