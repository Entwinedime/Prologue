"""Read SDK clocks, intervals, queue slots and event generations from native evidence."""

from __future__ import annotations

import json
import struct
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any

from prologue.common.io import read_jsonl
from prologue.common.observation import DeviceCalibration


class AscendClock(DeviceCalibration):
    def __init__(self, device: Path, samples: list[dict]) -> None:
        super().__init__(samples)
        physical = device.name.removeprefix("device_")
        if self.device_id != int(physical):
            raise ValueError("device clock calibration has a different device from the SDK profile")

        self.begin_ns, self.end_ns = self.midpoints[0], self.midpoints[-1]

        # Keep the epoch offset exact; float-to-string conversion can lose several device counter ticks.
        start = json.loads((device / f"start_info.{physical}").read_text())
        self.offset_us = Decimal(str(start["collectionTimeBegin"])) - Decimal(start["clockMonotonicRaw"]) / 1000
        info = json.loads((device / f"info.json.{physical}").read_text())
        self.frequency = Decimal(info["DeviceInfo"][0]["hwts_frequency"]) * 1_000_000
        host = self.values(device / f"host_start.log.{physical}")
        native = self.values(device / f"dev_start.log.{physical}")
        host_frequency = Decimal(info["CPU"][0]["Frequency"]) * 1_000_000
        self.host_anchor = host["clock_monotonic_raw"] + round(
            Decimal(host["cntvct_diff"]) * 1_000_000_000 / host_frequency
        )
        self.counter_anchor = native["cntvct"]

    @staticmethod
    def values(path: Path) -> dict[str, int]:
        return {key.strip(): int(value.strip()) for line in path.read_text().splitlines()
                if ":" in line for key, value in [line.split(":", 1)]}

    def timestamp(self, microseconds: Decimal) -> int:
        raw = (microseconds - self.offset_us) * 1000
        counter = self.counter_anchor + round((raw - self.host_anchor) * self.frequency / 1_000_000_000)

        # Extrapolated SDK setup/teardown records are filtered outside the valid observation window.
        return self.raw_timestamp(counter)

    def interval(self, record: dict) -> tuple[int, int]:
        begin = Decimal(str(record["ts"]))
        return self.timestamp(begin), self.timestamp(begin + Decimal(str(record["dur"])))


class HostCounter:
    """Convert SDK host measurements back to their native counter coordinates before relating them to RAW
    timestamps.
    """

    def __init__(self, profile: Path) -> None:
        start, = profile.glob("PROF_*/host/start_info")
        host, = profile.glob("PROF_*/host/host_start.log")
        device, = profile.glob("PROF_*/device_*")
        info = device / f"info.json.{device.name.removeprefix('device_')}"
        start = json.loads(start.read_text())
        anchor = AscendClock.values(host)
        info = json.loads(info.read_text())
        self.frequency = Decimal(info["CPU"][0]["Frequency"]) * 1_000_000
        self.anchor = anchor["cntvct"]
        self.framework_raw = anchor["clock_monotonic_raw"]
        self.cann_raw = self.framework_raw + round(Decimal(anchor["cntvct_diff"]) * 1_000_000_000 / self.frequency)
        self.offset = Decimal(start["collectionTimeBegin"]) * 1000 - int(start["clockMonotonicRaw"])

    def timestamp(self, counter: int) -> int:

        # The profiler records this counter/RAW anchor; recover time on its actual counter grid.
        return self.framework_raw + round(Decimal(counter - self.anchor) * 1_000_000_000 / self.frequency)

    def interval(self, record):

        # Preserve the export's integer wall/RAW offset; float epoch subtraction loses counter ticks.
        framework = record.get("cat") in {"cpu_op", "enqueue", "dequeue"}
        anchor = self.framework_raw if framework else self.cann_raw
        begin = Decimal(str(record["ts"])) * 1000
        end = begin + Decimal(str(record["dur"])) * 1000
        return tuple(
            self.anchor + round((value - self.offset - anchor) * self.frequency / 1_000_000_000)
            for value in (begin, end)
        )


HOST_WAITS = frozenset({
    "aclrtSynchronizeStream", "aclrtSynchronizeStreamWithTimeout", "aclrtSynchronizeEvent",
    "aclrtSynchronizeEventWithTimeout", "aclrtSynchronizeDeviceWithTimeout",
})


@dataclass
class Interval:
    begin: int
    end: int
    thread: int
    record: dict[str, Any]

    @classmethod
    def read(cls, record: dict[str, Any], counter: HostCounter) -> Interval:
        begin, end = counter.interval(record)
        return cls(begin, end, int(record["tid"]), record)

    def contains(self, other: Interval) -> bool:
        return self.thread == other.thread and self.begin <= other.begin and other.end <= self.end


class ThreadIntervals:
    """Index independently nested caller intervals with thread-local lookup coordinates."""

    def __init__(self, intervals: list[Interval]) -> None:
        self.threads: dict[int, tuple[list[int], list[Interval], list[int], list[int]]] = {}
        grouped: dict[int, list[Interval]] = defaultdict(list)
        for interval in intervals:
            grouped[interval.thread].append(interval)

        for thread, values in grouped.items():
            values.sort(key=lambda value: (value.begin, -value.end))
            parents = []
            furthest = []
            active = []

            for index, value in enumerate(values):
                while active and not values[active[-1]].contains(value):
                    active.pop()
                parents.append(active[-1] if active else -1)
                active.append(index)
                furthest.append(max(value.end, furthest[-1] if furthest else value.end))

            self.threads[thread] = [value.begin for value in values], values, parents, furthest

    def enclosing(self, interval: Interval) -> Interval | None:
        if interval.thread not in self.threads:
            return None
        starts, values, parents, _ = self.threads[interval.thread]
        index = bisect_right(starts, interval.begin) - 1
        while index >= 0:
            if values[index].contains(interval):
                return values[index]
            index = parents[index]
        return None

    def inside(self, interval: Interval) -> list[Interval]:
        if interval.thread not in self.threads:
            return []
        starts, values, _, _ = self.threads[interval.thread]
        begin = bisect_right(starts, interval.begin - 1)
        end = bisect_right(starts, interval.end)
        return [value for value in values[begin:end] if interval.contains(value)]

    def overlapping(self, interval: Interval) -> list[Interval]:
        if interval.thread not in self.threads:
            return []
        starts, values, _, furthest = self.threads[interval.thread]
        index = bisect_right(starts, interval.end) - 1
        result = []
        while index >= 0 and furthest[index] >= interval.begin:
            if values[index].end >= interval.begin:
                result.append(values[index])
            index -= 1
        return result


def task_flips(profile: Path) -> set[tuple[int, int, int]]:
    directory, = profile.glob("PROF_*/device_*/data")
    files = [path for path in directory.glob("ts_track.data.*.slice_*") if path.name.rsplit("_", 1)[1].isdigit()]
    files.sort(key=lambda path: int(path.name.rsplit("_", 1)[1]))
    data = b"".join(path.read_bytes() for path in files)
    record = struct.Struct("=BBHLQHHHH16B")

    # Match the SDK TsTrack parser's aged-prefix alignment and TS_TASK_FLIP layout.
    records = [row for row in record.iter_unpack(data[len(data) % record.size:]) if row[1] == 14]
    batches: dict[int, int] = defaultdict(int)
    flips = set()
    for row in sorted(records, key=lambda row: row[4]):
        stream, flip, task = row[5], row[6], row[8]

        # DeviceTaskCollector increments the epoch for every flip, including stream destruction (65535).
        batches[stream] += 1
        if flip != 65535:
            flips.add((stream, task, batches[stream]))

    return flips


def queue_marker(task, flips):
    return (task["Task Type"] == "PLACE_HOLDER_SQE"
            and (int(task["Physic Stream Id"]), int(task["Task Id"]), int(task["Batch Id"])) in flips)


def driver_submissions(runtime):
    queues = {}
    for native in sorted(runtime, key=lambda value: value.begin):
        row = native.record
        if not row["api"].startswith("halSq"):
            continue
        if row["result"] != 0:
            raise ValueError("recorded driver queue operation failed")

        key = row["device"], row["task_system"], row["queue"]
        if row["api"] == "halSqCqAllocate":
            queues[key] = row["stream"]
        elif row["api"] == "halSqCqFree":
            del queues[key]
        elif row["api"] == "halSqTaskSend":
            stream = queues[key]
            if type(row["tasks"]) is not int or row["tasks"] < 0:
                raise ValueError("driver queue posting requires an observed nonnegative SQE count")

            yield stream, native


STREAM_WAITS = frozenset({"aclrtStreamWaitEvent", "aclrtStreamWaitEventWithTimeout"})
EVENT_WAITS = STREAM_WAITS | {"aclrtSynchronizeEvent", "aclrtSynchronizeEventWithTimeout"}


def sdk_rows(path):
    return [row for row in json.loads(path.read_bytes(), parse_float=Decimal) if row.get("ph") == "X"]


def sdk_host_intervals(rows, counter):
    for row in rows:
        args = row.get("args", {})
        if "Task Type" not in args and (row.get("cat") in {"enqueue", "dequeue"} or "connection_id" in args):
            yield Interval.read(row, counter)


def native_interval(row):
    return Interval(row["begin_counter"], row["end_counter"], row["thread"], row)


def event_records(intervals, waits):
    records = {}
    for native in sorted(intervals, key=lambda interval: interval.begin):
        row = native.record
        if row["api"] == "aclrtRecordEvent":
            records[row["event"]] = native
        elif row["api"] == "aclrtDestroyEvent":
            records.pop(row["event"], None)
        elif row["api"] in waits:

            # Handles can be recorded again or destroyed between waits. Bind at the observation's native counter.
            yield native, records.get(row["event"]) if "event" in row else None


def device_streams(rows, clock, flips):
    streams = defaultdict(list)
    for row in rows:
        args = row.get("args", {})
        if "Task Type" in args and not queue_marker(args, flips):
            streams[int(args["Physic Stream Id"])].append(row)

    # A stream can reuse task numbers after a flip. Order by calibrated time, then the SDK's physical identity.
    for tasks in streams.values():
        tasks.sort(key=lambda row: (clock.interval(row)[0], *(row["args"][key]
                                    for key in ("Batch Id", "Task Id", "Subtask Id"))))
    return streams


def physical_slots(tasks, flips):
    streams = defaultdict(lambda: defaultdict(set))
    for task in tasks:
        slot = int(task["Batch Id"]), int(task["Task Id"])
        owner = None if queue_marker(task, flips) else task["connection_id"]
        streams[int(task["Physic Stream Id"])][slot].add(owner)

    # Physical slots include flips and may contain multiple SDK subtasks. Ownership proof belongs to the consumer.
    indexed = {}
    for stream, slots in streams.items():
        ordered = sorted(slots.items())
        positions = defaultdict(list)
        for index, (_, owners) in enumerate(ordered):
            for owner in owners:
                positions[owner].append(index)

        indexed[stream] = ordered, positions

    return indexed


class QueueSlots:
    """Prove release from one recording. Memoized groups depend only on its physical slots and driver posts."""

    def __init__(self, runtime, apis, rows, flips):
        self.apis, self.flips = apis, flips
        self.tasks = [row["args"] for row in rows if "Task Type" in row.get("args", {})]
        self.submissions = list(driver_submissions(runtime))
        by_caller, self.issued, self.post_index = defaultdict(list), defaultdict(list), {}

        for stream, post in self.submissions:
            by_caller[post.thread, stream].append(post)
            if post.record["tasks"]:
                self.post_index[id(post)] = len(self.issued[stream])
                self.issued[stream].append(post)

        self.by_caller = {key: ([p.begin for p in posts], posts) for key, posts in by_caller.items()}
        self.resolved = {}

    def inside(self, api, stream):
        starts, posts = self.by_caller.get((api.thread, stream), ((), []))
        contained = posts[bisect_left(starts, api.begin):bisect_right(starts, api.end)]
        if any(post.end > api.end for post in contained):
            raise ValueError("driver post escapes its recorded API interval")
        return contained

    @cached_property
    def slots(self):
        return physical_slots(self.tasks, self.flips)

    def posts(self, connection, stream):
        return [post for post in self.inside(self.apis[connection], stream) if post.record["tasks"]]

    def successive_posts(self, posts):
        return all(self.post_index[id(right)] == self.post_index[id(left)] + 1 and left.end <= right.begin
                   for left, right in pairwise(posts))

    @staticmethod
    def consecutive(slots):
        return all(
            (batch == following and successor == task + 1) or (following == batch + 1 and owners == {None})
            for ((batch, task), _), ((following, successor), owners) in pairwise(slots)
        )

    def neighbor(self, stream, position, post, side):
        issued = self.issued[stream]
        index = self.post_index[id(post)] + side
        slots, owners = self.slots[stream]

        if not 0 <= index < len(issued) or not 0 <= position < len(slots):
            return False
        tag = slots[position][1]
        if len(tag) != 1:
            return False

        connection, = tag
        if connection not in self.apis:
            return False
        group = self.posts(connection, stream)
        boundary = -1 if side < 0 else 0
        neighbor = issued[index]

        if not group or group[boundary] is not neighbor:
            return False
        if not self.successive_posts([neighbor, post] if side < 0 else [post, neighbor]):
            return False
        if len(group) == 1:
            return True

        tagged = owners[connection]
        block = slots[tagged[0]:tagged[-1] + 1]
        return (position == tagged[boundary] and len(block) == sum(p.record["tasks"] for p in group)
                and all(tags == tag for _, tags in block) and self.consecutive(block) and self.successive_posts(group))

    def prove_group(self, connection, stream, posts):

        # Each candidate interval consumes exactly the recorded SQE count, including proven batch flips.
        slots, by_owner = self.slots[stream]
        tagged = by_owner[connection]
        width = sum(post.record["tasks"] for post in posts)
        starts = []

        if posts and tagged and self.successive_posts(posts):
            for start in range(max(0, tagged[-1] - width + 1), min(tagged[0], len(slots) - width) + 1):
                end = start + width
                block = slots[start:end]
                if any(tags not in ({connection}, {None}) for _, tags in block) or not self.consecutive(block):
                    continue
                if width != len(tagged) and not (
                    self.neighbor(stream, start - 1, posts[0], -1)
                    and self.neighbor(stream, end, posts[-1], 1)
                    and self.consecutive(slots[start - 1:end + 1])
                ):
                    continue

                starts.append(start)

        if len(starts) != 1:
            raise ValueError(
                f"driver evidence cannot identify one release on stream {stream} for connection {connection}"
            )

        cursor, = starts
        releases = {}
        for post in posts:
            end = cursor + post.record["tasks"]
            for slot, tags in slots[cursor:end]:
                if connection in tags:
                    releases[slot] = post

            cursor = end

        return releases

    def release(self, task):
        stream, connection = int(task["Physic Stream Id"]), task["connection_id"]
        key = stream, connection
        if key not in self.resolved:
            posts = self.posts(connection, stream)
            if len(posts) == 1:
                return posts[0]
            self.resolved[key] = self.prove_group(connection, stream, posts)
        return self.resolved[key][int(task["Batch Id"]), int(task["Task Id"])]


class RawTraceIndex:
    """Index the SDK and native recorder independently for each projection or reference reader."""

    def __init__(self, profile, runtime, raw, rows, flips):
        self.cann, = profile.glob("PROF_*")
        device, = self.cann.glob("device_*")
        self.clock = AscendClock(device, raw["device_clock"])
        self.host_times = {}
        self.apis = {}
        counter = HostCounter(profile)
        queues = {"enqueue": [], "dequeue": []}

        for interval in sdk_host_intervals(rows, counter):
            row = interval.record
            for position in (interval.begin, interval.end):
                self.host_times[f"thread/{interval.thread}", position] = counter.timestamp(position)

            category = row.get("cat")
            if category in queues:
                queues[category].append(interval)
            elif "Thread Id" in row.get("args", {}) and "connection_id" in row["args"]:
                key = row["args"]["connection_id"]
                if key in self.apis:
                    raise ValueError(f"duplicate native API connection: {key}")

                self.apis[key] = interval

        self.enqueue = self.queue_index(queues["enqueue"])
        self.dequeue = self.queue_index(queues["dequeue"])
        self.enqueue_ranges = ThreadIntervals(queues["enqueue"])
        self.dequeue_ranges = ThreadIntervals(queues["dequeue"])
        self.api_ranges = ThreadIntervals(list(self.apis.values()))

        # Native samples override the SDK clock conversion at their own counter coordinates.
        markers, windows = raw["host_observation"], []
        self.signals = {i for i, point in enumerate(raw["points"]) if point["side"] == "signal"}
        self.scopes, self.points, self.excluded = {}, {}, []
        for row in markers["intervals"]:
            interval = self.native_interval(row)
            name = row["name"]
            if name == "prologue.window":
                windows.append(interval)
            elif name.startswith(("prologue.scope/", "prologue.point/")):
                prefix, number = name.split("/")
                identity = int(number)
                index = self.scopes if prefix == "prologue.scope" else self.points

                if identity in index:
                    raise ValueError(f"duplicate native marker: {name}")
                index[identity] = interval
                if index is self.points and identity not in self.signals:
                    self.excluded.append(interval)
            elif name in {"prologue.observer", "prologue.auxiliary"}:
                self.excluded.append(interval)

        if len(windows) != 1:
            raise ValueError("native recording requires one closed observation window")
        self.window, = windows

        # A closed service contributes its own host interval, outside the physical marker owner index.
        self.marker_ranges = ThreadIntervals(list(self.scopes.values()))
        self.closed_scopes = set()
        for scope in raw["scopes"]:
            identity = scope["id"]
            if "service" in scope:
                if identity in self.scopes:
                    raise ValueError(f"service has two competing host observations: {identity}")
                self.scopes[identity] = self.native_interval({
                    **scope["host_interval"], "thread": scope["thread"], "name": f"prologue.scope/{identity}",
                })

            if "service" in scope or scope["parent"] in self.closed_scopes:
                self.closed_scopes.add(identity)

        if set(self.scopes) != set(range(1, len(raw["scopes"]))):
            raise ValueError("native scope metadata and recorded CPU markers differ")
        self.scope_ranges = ThreadIntervals(list(self.scopes.values()))
        self.excluded_ranges = ThreadIntervals(self.excluded)

        self.runtime = []
        for row in read_jsonl(runtime / f"{markers['process']}.jsonl"):
            if row["result"] != 0:
                raise RuntimeError(f"native runtime call failed: {row['api']} ({row['result']})")
            self.runtime.append(self.native_interval(row))

        self.runtime_ranges = ThreadIntervals(self.runtime)
        self.queues = QueueSlots(self.runtime, self.apis, rows, flips)

    def native_interval(self, record: dict) -> Interval:
        thread = record["thread"]
        for side in ("begin", "end"):
            for resource in (f"thread/{thread}", f"service/{thread}"):
                self.host_times[resource, record[f"{side}_counter"]] = record[f"{side}_raw_ns"]
        return native_interval(record)

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

    def native_api(self, native: Interval) -> Interval:
        calls = [api for api in self.api_ranges.inside(native) if api.record["args"]["id"] == native.record["api"]]
        if len(calls) != 1:
            raise ValueError(f"native call does not identify one profiler API: {native.record['api']}")
        return calls[0]

    def reserved_host_intervals(self, repeated_host_waits):
        for interval in self.runtime:
            call = interval.record
            identity = call["thread"], call["begin_counter"]
            if identity in repeated_host_waits:
                continue
            if call["api"] in HOST_WAITS or (call["api"] == "aclrtMemcpy" and call["bytes"]):
                yield call["thread"], call["begin_raw_ns"], call["end_raw_ns"]

    def point_api(self, identity):
        marker = self.points[identity]
        queues = self.enqueue_ranges.inside(marker)
        if len(queues) != 1:
            raise ValueError(f"device point {identity} does not have one native enqueue")
        dequeue = self.dequeue[queues[0].record["args"]["correlation_id"]]
        calls = [api for api in self.api_ranges.inside(dequeue) if api.record["args"]["id"] == "aclrtRecordEvent"]
        if len(calls) != 1:
            raise ValueError(f"device point {identity} does not identify one record API")
        return calls[0]

    def host_interval(self, interval):
        resource = f"thread/{interval.thread}"
        return interval.thread, self.host_times[resource, interval.begin], self.host_times[resource, interval.end]

    def dequeued_intervals(self):
        for identity, dequeue in self.dequeue.items():
            observer_owned = self.excluded_ranges.enclosing(self.enqueue[identity]) is not None
            yield self.host_interval(dequeue), observer_owned
