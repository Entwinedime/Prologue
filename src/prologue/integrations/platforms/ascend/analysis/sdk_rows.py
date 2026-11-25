"""Decode raw SDK coordinates, queue markers and driver records without assigning graph work."""

from __future__ import annotations

import struct
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clock import HostCounter

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
