"""Recover SDK host coordinates and RAW time, and calibrate measured device time."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from prologue.common.observation.device_clock import DeviceCalibration


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
