"""Calibrate a device counter against bracketed host clock samples."""

from bisect import bisect_right
from fractions import Fraction
from math import ceil, floor


class DeviceCalibration:
    """Map device counter ticks between independently bracketed RAW snapshots."""

    def __init__(self, samples: list[dict]) -> None:
        if len(samples) < 2:
            raise ValueError("device clock requires at least two independent samples")
        self.device_id = samples[0]["device"]
        if any(
            sample["device"] != self.device_id or sample["begin_raw_ns"] > sample["end_raw_ns"] for sample in samples
        ) or any(
            left["counter"] >= right["counter"] or left["end_raw_ns"] >= right["begin_raw_ns"]
            for left, right in zip(samples, samples[1:], strict=False)
        ):
            raise ValueError("device clock requires advancing, separated calibration intervals from one device")
        self.samples = samples
        self.counters = [sample["counter"] for sample in samples]
        self.midpoints = [(sample["begin_raw_ns"] + sample["end_raw_ns"]) // 2 for sample in samples]

    def counter_segment(self, counter: int) -> int:
        return min(max(bisect_right(self.counters, counter) - 1, 0), len(self.counters) - 2)

    def raw_timestamp(self, counter: int) -> int:
        index = self.counter_segment(counter)
        begin, end = self.counters[index:index + 2]
        start, finish = self.midpoints[index:index + 2]
        return start + (counter - begin) * (finish - start) // (end - begin)

    def counter_at(self, timestamp: int) -> tuple[int, int]:
        index = min(max(bisect_right(self.midpoints, timestamp) - 1, 0), len(self.midpoints) - 2)
        start, finish = self.midpoints[index:index + 2]
        begin, end = self.counters[index:index + 2]
        counts, duration = end - begin, finish - start

        # Invert the integer floor exactly so a recorded endpoint identifies one physical tick.
        first = -((start - timestamp) * counts // duration)
        last = -((start - timestamp - 1) * counts // duration) - 1
        if first != last or not 0 <= first <= counts:
            raise ValueError("device timestamp does not identify one counter inside its sampled clock segment")
        return index, begin + first

    def bounds(self, timestamp: int) -> tuple[int, int]:
        index, counter = self.counter_at(timestamp)
        first, last = self.samples[index:index + 2]
        offset, counts = counter - first["counter"], last["counter"] - first["counter"]
        start, finish = first["begin_raw_ns"], last["begin_raw_ns"]
        lower = start + offset * (finish - start) // counts
        start, finish = first["end_raw_ns"], last["end_raw_ns"]
        upper = start - (offset * (start - finish) // counts)
        return lower, upper

    def elapsed_bounds(self, start: int, finish: int) -> tuple[int, int]:
        weights = {}
        for time, sign in ((start, -1), (finish, 1)):
            index, counter = self.counter_at(time)
            first, last = self.counters[index:index + 2]
            part = Fraction(counter - first, last - first)
            for knot, weight in ((index, 1 - part), (index + 1, part)):
                weights[knot] = weights.get(knot, 0) + sign * weight

        # The two endpoints share clock knots. Their common offset cancels in a local duration.
        lower = upper = 0
        for knot, weight in weights.items():
            sample = self.samples[knot]
            a, b = weight * sample["begin_raw_ns"], weight * sample["end_raw_ns"]
            lower += min(a, b)
            upper += max(a, b)
        return floor(lower), ceil(upper)
