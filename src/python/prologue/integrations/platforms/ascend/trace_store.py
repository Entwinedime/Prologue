"""Read SDK clocks, intervals, queue slots and event generations from native evidence."""

from __future__ import annotations

import json
import struct
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
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


def sdk_intervals(rows, host, device=None):
    for row in rows:
        args = row.get("args", {})
        if row.get("cat") not in {"enqueue", "dequeue"} and "connection_id" not in args:
            continue
        if "Task Type" not in args:
            yield Interval.read(row, host)
        elif device is not None:
            yield Interval.read(row, device)


def runtime_rows(directory, process):
    return read_jsonl(directory / f"{process}.jsonl")


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


def device_tasks(rows):
    return [row["args"] for row in rows if "Task Type" in row.get("args", {})]


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
