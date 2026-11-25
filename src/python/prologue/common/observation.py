"""Record semantic calls, worker observations and calibrated times."""

from __future__ import annotations

import threading
from bisect import bisect_right
from collections import defaultdict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import ceil, floor
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Parameter:
    scope: int
    name: str
    member: int | None = None


class ScopeParameters:
    def __init__(self, scopes) -> None:
        self.scopes = scopes

    def resolve(self, expression, scope, member=None):
        if isinstance(expression, int):
            return expression
        if isinstance(expression, str):
            if member is not None:
                if expression not in ("context_tokens", "new_tokens"):
                    raise ValueError(f"unknown admitted member feature: {expression}")
                return Parameter(scope, expression, member)
            record = self.scopes[scope]
            if expression in record["constants"]:
                return record["constants"][expression]
            if expression in record["arguments"]:
                return self.resolve(
                    record["arguments"][expression], record["parent"], record.get("row_call", {}).get("member")
                )
            if expression not in record["parameters"]:
                raise ValueError(f"shape references undeclared parameter: {scope}:{expression}")
            return Parameter(scope, expression)
        return expression[0], *(self.resolve(value, scope, member) for value in expression[1:])

    def bind(self, rule, scope):
        return tuple(self.resolve(axis, scope) for axis in rule)

    def local(self, rule, scope):
        names = {}
        for name in self.scopes[scope]["parameters"]:
            expression = self.resolve(name, scope)

            # A literal dimension must not become a variable merely because its observed value matches.
            if not isinstance(expression, int):
                names.setdefault(expression, name)
        return [self.render(axis, names) for axis in rule]

    def render(self, expression, names):
        if isinstance(expression, int):
            return expression
        if expression in names:
            return names[expression]
        if isinstance(expression, Parameter):
            raise ValueError(f"producer shape parameter is not bound in the consumer: {expression}")
        return [expression[0], *(self.render(value, names) for value in expression[1:])]


class CapturedCall:
    def __init__(self, capture: SemanticCapture, record: dict[str, Any], device: bool, stream: Any) -> None:
        self.capture = capture
        self.record = record
        self.identity = record["id"]
        self.device = device
        self.stream = stream

    def __enter__(self) -> CapturedCall:
        self.parent: Token = self.capture.current.set(self.identity)
        try:
            if self.device:
                self.record["device_entry"] = self.capture.runtime.point(self.identity, "entry", self.stream)
            if "service" in self.record:
                self.record["host_interval"] = {}
                self.native = self.capture.runtime.service_scope(self.record["host_interval"])
            else:
                self.native = self.capture.runtime.scope(f"prologue.scope/{self.identity}")
            self.native.__enter__()
        except BaseException:
            self.capture.current.reset(self.parent)
            raise
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        try:
            self.native.__exit__(kind, error, traceback)
            if error is None and self.device:
                self.record["device_return"] = self.capture.runtime.point(self.identity, "return", self.stream)
            self.record["complete"] = error is None
        finally:
            self.capture.current.reset(self.parent)

    def port(
        self, name: str, direction: str, tensor: dict[str, Any] | None, *, use: str, at_return: bool = False,
    ) -> None:
        if direction not in {"input", "output"} or use not in {"host", "device", "metadata", "state"}:
            raise ValueError("a boundary port needs a direction and an explicit use")
        port = {"name": name, "direction": direction, "tensor": tensor, "use": use}
        if at_return:
            if direction != "input" or use != "host" or tensor is None:
                raise ValueError("a completed read requires a host tensor input")
            port["at_return"] = True
        self.record["ports"].append(port)

    def output(self, name: str, tensor: dict[str, Any], *, use: str, preceding: list[int]) -> None:
        writer = tensor["writer"]
        if writer in preceding and not any(
            port["direction"] == "input" and port["use"] in {"host", "device"} and port["tensor"] is not None
            and (port["tensor"]["storage"], port["tensor"]["version"]) == (tensor["storage"], tensor["version"])
            for port in self.record["ports"]
        ):

            # Deferred values need their actual producer; an explicit continuation input already supplies it.
            if tensor["allocation"] != writer:
                raise ValueError("a deferred mutation needs an explicit input/output boundary at its writer")
            self.capture.scopes[writer]["ports"].append({
                "name": name, "direction": "output", "tensor": tensor, "use": use,
            })
            self.port(name, "input", tensor, use=use)
        self.port(name, "output", tensor, use=use)

    def signal(self, name: str, direction: str, point: int) -> None:
        if direction not in {"input", "output"}:
            raise ValueError("a completion signal needs an input or output direction")
        self.record["ports"].append({"name": name, "direction": direction, "use": "signal", "point": point})

    def work_signal(self, name: str, direction: str, producer: int) -> None:
        if direction not in {"input", "output"}:
            raise ValueError("a work completion needs an input or output direction")

        # The runtime resolves the Work's actual end event from its native wait, not from Python return time.
        self.record["ports"].append({"name": name, "direction": direction, "use": "signal", "producer": producer})


class SemanticCapture:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.current: ContextVar[int] = ContextVar("semantic_call", default=0)
        self.lock = threading.Lock()
        self.closed = False
        self.scopes: list[dict[str, Any]] = [{
            "id": 0, "parent": None, "role": "window", "site": "", "implementation": "",
            "parameters": {}, "constants": {}, "arguments": {}, "work": {}, "bindings": {},
            "ports": [], "complete": True,
        }]

    def call(
        self, role: str, site: str, implementation: str, *,
        parameters: dict[str, int] | None = None,
        constants: dict[str, int] | None = None,
        arguments: dict[str, Any] | None = None,
        work: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
        device: bool = False,
        stream: Any = None,
        parent: int | None = None,
        service: dict[str, Any] | None = None,
    ) -> CapturedCall:
        parent = self.current.get() if parent is None else parent
        enclosing = self.scopes[parent]

        # Inheritance is by declared parameter name; no observed dimension is matched by its numeric value.
        inherited = parameters is None
        if constants is None:
            fixed = dict(enclosing["constants"]) if inherited else {}
        else:
            fixed = dict(constants)
        supplied = dict(enclosing["parameters"] if inherited else parameters)
        for name, value in fixed.items():
            if name in supplied and supplied[name] != value:
                raise ValueError(f"fixed parameter differs from the native call: {site}:{name}")
            supplied[name] = value
        relations = {} if arguments is None else dict(arguments)
        if inherited and arguments is None:
            relations = {name: name for name in supplied if name not in fixed}
        if any(not isinstance(value, int) or isinstance(value, bool) for value in supplied.values()):
            raise TypeError("semantic parameters must be integers")

        with self.lock:
            if self.closed:
                raise RuntimeError("semantic observation is already closed")
            record = {
                "id": len(self.scopes), "parent": parent, "role": role, "site": site,
                "implementation": implementation, "parameters": supplied, "constants": fixed,
                "arguments": relations, "work": dict(enclosing["work"] if work is None else work),
                "bindings": {} if bindings is None else dict(bindings), "ports": [], "complete": False,
                "thread": threading.get_native_id(),
            }
            if service is not None:
                record["service"] = service
            self.scopes.append(record)
        return CapturedCall(self, record, device, stream)

    def document(self, environment: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.current.get() != 0 or any(not scope["complete"] for scope in self.scopes):
                raise RuntimeError("native observation contains an unfinished semantic call")
            self.closed = True
            return {"environment": environment, "scopes": self.scopes, "points": self.runtime.points}


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


def merged(ranges):
    result = []
    for begin, end in sorted(ranges):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((begin, end))
    return result


class ThreadService:
    def __init__(self, rows, dequeues, excluded, unavailable=()):
        self.activity = {}
        self.excluded = self.by_thread(excluded)
        self.unavailable = self.by_thread(unavailable)
        self.dequeues = self.by_thread(dequeues)
        self.loss = []
        for row in rows:
            thread = row["thread"]
            switches = row["switches"]
            if row["lost"] or any(left[0] > right[0] or left[1] == right[1] for left, right in pairwise(switches)):
                self.loss.append(thread)
                continue
            if not switches:
                continue
            begin, end = row["window_ns"]
            running = not switches[0][1]
            cursor = begin
            intervals = []
            for stamp, next_state in switches:
                if stamp > end:
                    break
                if stamp >= begin:
                    if running:
                        intervals.append((cursor, stamp))
                    cursor = stamp
                running = next_state
            if running:
                intervals.append((cursor, end))
            self.activity[thread] = (begin, end, intervals)

    @staticmethod
    def by_thread(intervals):
        ranges = defaultdict(list)
        for thread, begin, end in intervals:
            ranges[thread].append((begin, end))
        return {thread: merged(values) for thread, values in ranges.items()}

    @staticmethod
    def intersect(first, second):
        left, right = iter(first), iter(second)
        a, b = next(left, None), next(right, None)
        while a is not None and b is not None:
            begin, end = max(a[0], b[0]), min(a[1], b[1])
            if begin < end:
                yield begin, end
            if a[1] <= b[1]:
                a = next(left, None)
            else:
                b = next(right, None)

    @staticmethod
    def within(intervals, begin, end):
        first = bisect_right(intervals, begin, key=lambda interval: interval[1])
        last = bisect_right(intervals, end - 1, lo=first, key=lambda interval: interval[0])
        return intervals[first:last]

    def running_intervals(self, thread, begin, end):
        if thread not in self.activity or end <= begin:
            return None
        opened, closed, active = self.activity[thread]
        if not opened <= begin < end <= closed:
            return None
        return self.intersect(self.within(active, begin, end), [(begin, end)])

    def duration(self, thread, begin, end, submission):
        if self.within(self.unavailable.get(thread, ()), begin, end):
            return None
        covered = self.running_intervals(thread, begin, end)
        if covered is None:
            return None
        if submission:

            # An idle or spinning submission thread between two queue commands performs no modeled work.
            covered = self.intersect(covered, self.within(self.dequeues.get(thread, ()), begin, end))
        covered = list(covered)
        occupied = sum(y - x for x, y in covered)
        excluded = sum(
            y - x for x, y in self.intersect(covered, self.within(self.excluded.get(thread, ()), begin, end))
        )
        return occupied - excluded


@dataclass(frozen=True, slots=True)
class ProfileWindow:
    observed: tuple[int, ...]
    replay: bool = False

    def protocol_directory(self, capture: Path) -> Path:
        root = capture / "protocol"
        return root / f"worker-{self.observed[0]}" if self.replay else root


class CapturedWorkers:
    def __init__(self, instances):
        self.owners = {}
        self.observed = []
        self.windows = {}
        self.joint_execution = True

        for instance in instances:
            owner = instance["id"]
            participants = instance["native_participants"]
            observed = instance["observed_roles"]
            if not isinstance(owner, str) or not owner or owner in self.windows or not participants:
                raise ValueError("a capture needs unique instances with participating workers")
            for rank in participants:
                if type(rank) is not int or rank < 0 or rank in self.owners:
                    raise ValueError("native participants must be distinct nonnegative integer workers")
                self.owners[rank] = owner
            if (any(type(rank) is not int or rank not in participants for rank in observed)
                    or len(set(observed)) != len(observed)):
                raise ValueError("observed workers must be distinct participants of their own instance")
            self.observed.extend(observed)

            # Placement qualifies peer timing. Replay windows identify which protocol records accompany a profile.
            placement = instance.get("placement", "full")
            if placement not in {"full", "single_device"}:
                raise ValueError(f"unknown worker placement: {placement}")
            replays = instance.get("profile_windows")
            if replays is None:
                windows = [ProfileWindow(tuple(observed))] if observed else []
            else:
                if (any(len(row) != 1 or type(row[0]) is not int for row in replays)
                        or sorted(row[0] for row in replays) != sorted(observed)):
                    raise ValueError("worker replay windows must cover each profiled worker exactly once")
                windows = [ProfileWindow(tuple(row), replay=True) for row in replays]
            self.windows[owner] = windows
            if placement != "full" or replays is not None:
                self.joint_execution = False

        if not self.owners:
            raise ValueError("a native capture requires participating workers")
