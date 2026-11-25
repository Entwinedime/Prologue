"""Resolve stream command coordinates and scoped native sampling intervals."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import write_json
from prologue.integrations.platforms.ascend.analysis.trace import AscendTrace, DeviceTask
from prologue.integrations.platforms.ascend.runtime.sampling import AscendExport
from prologue.integrations.platforms.ascend.trace_store import (
    HostCounter,
    Interval,
    ThreadIntervals,
    queue_marker,
    sdk_rows,
    task_flips,
)


def span_references(result):
    for edge in result["dependencies"]:
        yield edge["source"]
        yield edge["target"]

    for scope in result["scopes"]:
        for port in scope["ports"]:
            yield port["at"]
            if "wait" in port:
                yield port["wait"]["begin"]
            if "rendezvous" in port:
                yield port["rendezvous"]

        for field in ("kv_accesses", "slot_accesses"):
            for access in scope[field]:
                yield access["start"]
                yield access["finish"]


@dataclass
class NativeStreamCall:
    api: Interval
    stream: int
    scope: int
    native: Interval
    completion: Interval | None = None


@dataclass
class StreamCommand:
    source: DeviceTask | NativeStreamCall
    begin: int
    end: int
    post: Interval | None = None

    @property
    def stream(self) -> int:
        return self.source.stream


class CommandPositions:
    def __init__(self, trace: AscendTrace, tasks: list[DeviceTask]) -> None:
        bounds = [(api.end, api.begin, index) for index, task in enumerate(tasks)
                  if (api := trace.apis.get(task.record["args"]["connection_id"])) is not None]

        # CPU API boundaries constrain insertion, while the stored indices retain native device order.
        endings = sorted((end, index + 1) for end, _, index in bounds)
        self.before, self.latest = [end for end, _ in endings], [0]
        for _, position in endings:
            self.latest.append(max(self.latest[-1], position))

        beginnings = sorted((begin, end, index) for end, begin, index in bounds)
        self.after = [(begin, end) for begin, end, _ in beginnings]
        self.earliest = [len(tasks)] * (len(beginnings) + 1)
        for index in range(len(beginnings) - 1, -1, -1):
            self.earliest[index] = min(self.earliest[index + 1], beginnings[index][2])

    def bounds(self, api: Interval) -> tuple[int, int]:
        lower = self.latest[bisect_right(self.before, api.begin)]

        # The end-before branch wins when both APIs have zero duration at the same time.
        upper = self.earliest[bisect_right(self.after, (api.end, api.begin))]
        return lower, upper


class StreamCommands:
    def __init__(self, trace: AscendTrace) -> None:
        self.trace = trace
        self.commands: list[StreamCommand] = []
        self.stream_calls: dict[int, StreamCommand] = {}
        self.physical: dict[tuple[int, int], StreamCommand] = {}
        self.native_calls: list[NativeStreamCall] = []
        self.effects: dict[int, list[StreamCommand]] = {}
        self.order_commands()
        self.payload = [command for command in self.commands if isinstance(command.source, DeviceTask)]

    def task(self, task: DeviceTask) -> StreamCommand:
        return self.physical[task.stream, task.begin]

    def order_commands(self) -> None:
        trace = self.trace
        calls: dict[int, list[NativeStreamCall]] = defaultdict(list)
        connections = set()
        covered = {}
        for native in trace.runtime:
            name = native.record["api"]
            copy = name == "aclrtMemcpyAsync" and native.record["bytes"] > 0
            if name not in trace.stream_waits and not copy:
                continue
            if native.end <= trace.window.begin or trace.window.end <= native.begin:
                continue
            api = trace.native_api(native)
            connection = api.record["args"]["connection_id"]

            origin = trace.origin(api)
            owner = trace.owner(origin)
            if owner is None or owner in trace.closed_scopes or trace.excluded_ranges.enclosing(origin) is not None:
                continue
            if native.record["stream_status"] != 0:
                raise ValueError("native stream operation has no observed stream identity")
            call = NativeStreamCall(api, native.record["stream"], owner, native)
            self.native_calls.append(call)
            if copy:
                tasks = trace.tasks.get(connection) or self.covering_group(call)
                if tasks:
                    if any(task.stream != call.stream for task in tasks):
                        raise ValueError("native copy differs from its SDK command stream")
                    covered[connection] = tasks
                    continue

                call.completion = self.copy_completion(call)
                self.effects[connection] = []
                continue

            # Only actual queue operations receive stream coordinates; synchronous copies remain on their caller.
            calls[call.stream].append(call)
            connections.add(connection)

        streams: dict[int, list[DeviceTask]] = defaultdict(list)
        for task in trace.device_tasks:
            if task.record["args"]["connection_id"] not in connections:
                streams[task.stream].append(task)
        for stream in sorted(streams.keys() | calls.keys()):
            tasks = sorted(streams[stream], key=lambda task: task.begin)
            positions = [task.begin for task in tasks]
            command_positions = None
            gaps: dict[int, list[NativeStreamCall]] = defaultdict(list)
            for call in calls[stream]:
                observed = trace.tasks.get(call.api.record["args"]["connection_id"], [])
                if observed:
                    task, = observed
                    if task.stream != stream or task.record["args"]["Task Type"] != "EVENT_WAIT":
                        raise ValueError("stream wait differs from its SDK device task")
                    position = bisect_left(positions, task.begin)
                else:

                    # A missing SDK task is a logical wait only when the native API submitted no physical queue work.
                    if any(submission.record["tasks"] != 0 for submission in trace.queues.inside(call.native, stream)):
                        raise ValueError("unmatched stream wait has native queue submissions")

                    # Submission order bounds the missing command; concurrent commands cannot be guessed past.
                    if command_positions is None:
                        command_positions = CommandPositions(trace, tasks)
                    lower, upper = command_positions.bounds(call.api)
                    if lower != upper:
                        raise ValueError(
                            f"stream {stream} operation in scope {call.scope} has no unique "
                            f"command position: {lower}..{upper}"
                        )
                    position = lower
                gaps[position].append(call)

            ordered = []
            for position in range(len(tasks) + 1):
                local = gaps[position]
                if all(trace.tasks.get(call.api.record["args"]["connection_id"]) for call in local):
                    local.sort(key=lambda call: trace.tasks[call.api.record["args"]["connection_id"]][0].begin)
                else:
                    local.sort(key=lambda call: call.api.begin)
                    if any(left.api.end > right.api.begin for left, right in zip(local, local[1:], strict=False)):
                        raise ValueError(
                            f"concurrent native stream operations have no established order on stream {stream}"
                        )
                ordered.extend(local)
                if position < len(tasks):
                    ordered.append(tasks[position])

            # Derived ordinals never overwrite SDK task positions or independently measured timestamps.
            for index, source in enumerate(ordered):
                command = StreamCommand(source, 2 * index, 2 * index + 1)
                self.commands.append(command)
                physical = (
                    [source]
                    if isinstance(source, DeviceTask)
                    else trace.tasks.get(source.api.record["args"]["connection_id"], [])
                )
                for task in physical:
                    self.physical[task.stream, task.begin] = command
                if isinstance(source, NativeStreamCall):
                    connection = source.api.record["args"]["connection_id"]
                    self.stream_calls[connection] = command
                    self.effects[connection] = [command]

        for connection, tasks in covered.items():
            self.effects[connection] = [self.task(task) for task in tasks]

    def copy_completion(self, source: NativeStreamCall) -> Interval:
        row = source.native.record
        copies = [call for call in self.trace.runtime_ranges.inside(source.native)
                  if call.record["api"] in {"drvMemcpy", "halMemcpy"}
                  and all(call.record[key] == row[key] for key in ("source", "destination", "bytes"))]
        if not copies:
            raise ValueError(f"copy in scope {source.scope} "
                             "has neither correlated device work nor an observed synchronous transfer of its buffers")

        # Staging copies have different addresses. A successful synchronous copy of the full range proves visibility.
        return max(copies, key=lambda call: call.end)

    def covering_group(self, call: NativeStreamCall) -> list[DeviceTask]:
        trace = self.trace
        submissions = trace.queues.inside(call.native, call.stream)
        if not submissions:
            return []
        groups = []
        for api in trace.api_ranges.overlapping(call.api):
            tasks = trace.tasks.get(api.record["args"]["connection_id"], [])
            if not tasks or not api.contains(call.api) or trace.owner(trace.origin(api)) != call.scope:
                continue
            if (
                all(task.stream == call.stream for task in tasks)
                and trace.queues.inside(api, call.stream) == submissions
            ):
                groups.append(tasks)
        if len(groups) > 1:
            raise ValueError(f"native copy in scope {call.scope} has competing SDK command groups")

        # The same driver submissions observed through nested APIs represent one group, not extra work.
        return groups[0] if groups else []

    def posting(self, command: StreamCommand) -> tuple[Interval, Interval]:
        trace = self.trace
        source = command.source
        if isinstance(source, NativeStreamCall):
            api = source.api
            if command.post is None:
                physical = trace.tasks.get(api.record["args"]["connection_id"])

                # Elided waits retain their host condition with no physical queue service.
                command.post = trace.queues.release(physical[0].record["args"]) if physical else source.native
        else:
            arguments = source.record["args"]
            api = trace.apis[arguments["connection_id"]]
            if command.post is None:
                command.post = trace.queues.release(arguments)
        return api, command.post

    def submission_frontiers(self, staged_boundaries) -> None:
        trace = self.trace
        groups = {}
        for command in self.commands:
            source = command.source
            if isinstance(source, DeviceTask):
                connection = source.record["args"]["connection_id"]
                if source.service is not None or connection not in trace.apis or trace.outside_window(source):
                    continue
                observing = connection in trace.observer_connections or trace.auxiliary(source)
                if observing and (command.stream, command.end) not in staged_boundaries:
                    continue
            _, post = self.posting(command)
            groups[command.stream, post.thread, post.begin] = command

        # Only one frontier is visible for the tasks released by each physical post.
        submitted: dict[int, list[tuple[int, StreamCommand]]] = defaultdict(list)
        submitting = []
        for command in groups.values():
            post = command.post
            submitting.append(Interval(post.begin, post.end, command.stream, {}))
            submitted[command.stream].append((post.end, command))
        self.submitting = ThreadIntervals(submitting)
        self.frontiers = {}
        for stream, values in submitted.items():
            values.sort(key=lambda value: value[0])
            latest = []
            for _, command in values:
                latest.append(command if not latest or command.end > latest[-1].end else latest[-1])
            self.frontiers[stream] = [time for time, _ in values], latest

    def submitted(self, before: int, stream: int | None) -> list[StreamCommand]:
        result = []
        if stream is not None and stream not in self.frontiers:

            # Marker-only streams contain no business work after the drained warmup.
            return result
        selected = self.frontiers if stream is None else {stream: self.frontiers[stream]}
        for stream, (starts, latest) in selected.items():
            if any(
                window.begin < before < window.end
                for window in self.submitting.overlapping(Interval(before, before, stream, {}))
            ):
                raise ValueError(f"stream {stream} fence overlaps a native queue submission")
            index = bisect_right(starts, before) - 1
            if index >= 0:
                result.append(latest[index])
        return result

    def physical_record(self, command: StreamCommand) -> dict | None:
        source = command.source
        if isinstance(source, DeviceTask):
            return source.record
        tasks = self.trace.tasks.get(source.api.record["args"]["connection_id"])
        return tasks[0].record if tasks else None

    def evidence(self, resources: dict) -> list[dict]:
        records = []
        for call in self.native_calls:
            connection = call.api.record["args"]["connection_id"]
            commands = []
            for command in self.effects[connection]:
                record = self.physical_record(command)
                task = record["args"] if record else None
                commands.append(
                    {
                        "begin": command.begin,
                        "end": command.end,
                        "sdk": {name: task[name] for name in ("connection_id", "Batch Id", "Task Id", "Subtask Id")}
                        if task
                        else None,
                    }
                )
            records.append(
                {
                    "thread": call.native.thread,
                    "begin_counter": call.native.begin,
                    "connection": connection,
                    "source_scope": call.scope,
                    "resource": resources["thread", call.completion.thread]
                    if call.completion
                    else resources["stream", call.stream],
                    "commands": commands,
                    "completion": {key: call.completion.record[key] for key in ("api", "thread", "begin_counter")}
                    if call.completion
                    else None,
                }
            )
        return records

    def measurements(self, observation: dict) -> dict:
        trace = self.trace
        spans, resources = observation["spans"], observation["resources"]
        positions = [set() for _ in resources]
        for span in spans:
            positions[span["resource"]].update((span["begin"], span["end"]))
        for anchor in span_references(observation):
            positions[spans[anchor["span"]]["resource"]].add(anchor["order"])

        clocks = {}
        for command in self.commands:
            record = self.physical_record(command)
            interval = Interval.read(record, trace.clock) if record else None

            # Only real SDK tasks provide device timestamps. Unmeasured waits remain pure boundaries.
            clocks[command.stream, command.begin] = interval.begin if interval else None
            clocks[command.stream, command.end] = interval.end if interval else None

        external = {span["resource"] for span in spans
                    if "protocol_operand" in span and resources[span["resource"]]["kind"] == "service"}
        coordinates, device_services = [], []
        for resource, orders in enumerate(positions):
            if resource in external:

                # Staging proves the protocol boundaries. Only physical transport can measure its service.
                coordinates.extend({"resource": resource, "order": order, "time_ns": None} for order in sorted(orders))
                continue
            descriptor = resources[resource]
            stream = int(descriptor["identity"].split("/")[1]) if descriptor["kind"] == "stream" else None
            for order in sorted(orders):

                # Every retained host cut names an observed native or SDK endpoint; missing handoff is an error.
                measured = (
                    clocks[stream, order] if stream is not None else trace.host_times[descriptor["identity"], order]
                )
                if (
                    stream is not None
                    and measured is not None
                    and not trace.clock.begin_ns <= measured <= trace.clock.end_ns
                ):
                    raise ValueError("observed device boundary lies outside the clock calibration window")
                coordinates.append({"resource": resource, "order": order, "time_ns": measured})
            if stream is not None:

                # Only physical payload contributes service time. Gaps between commands remain timeline evidence.
                for begin, end in descriptor["payload"]:
                    start, finish = clocks[stream, begin], clocks[stream, end]
                    if start is None or finish is None or finish < start:
                        raise ValueError("native device payload has no ordered execution measurements")
                    device_services.append([resource, begin, end, finish - start])
        return {
            "clock": "monotonic_raw_ns",
            "window_ns": [
                trace.host_times[f"thread/{trace.window.thread}", side]
                for side in (trace.window.begin, trace.window.end)
            ],
            "coordinates": coordinates,
            "device_services": device_services,
        }


def tensor_layouts(environment):
    if not environment["device"].startswith("Ascend"):
        return set()
    layouts = {"positions"}
    if environment["implementation"]["attention_backend"] == "ascend":
        layouts.add("packed_value")
    return layouts


class ScopedTasks:
    @staticmethod
    def profile(root, rank):
        import torch_npu.profiler as sdk

        return sdk.profile(
            activities=[sdk.ProfilerActivity.CPU, sdk.ProfilerActivity.NPU],
            on_trace_ready=sdk.tensorboard_trace_handler(str(root / "native" / f"rank-{rank}"), analyse_flag=False),
            experimental_config=sdk._ExperimentalConfig(profiler_level=sdk.ProfilerLevel.Level1,
                                                        aic_metrics=sdk.AiCMetrics.AiCoreNone),
        )

    @staticmethod
    def identity(row):
        return row["args"]["Task Type"], row["name"]

    @classmethod
    def source(cls, capture, trace, rank, catalog, scopes, facts):
        native = json.loads((capture / f"rank-{rank}.json").read_text())
        profile, = (capture / "native" / f"rank-{rank}").glob("*_ascend_pt")
        source = AscendTrace(profile, capture / "runtime", native, trace)
        tasks = {(command.stream, command.begin, command.end): cls.identity(command.source.record)
                 for command in StreamCommands(source).payload}
        graph = catalog.graph
        if graph["evidence"]["window_ns"] != facts["measurements"]["window_ns"]:
            raise ValueError("source task partitions and template work use different observation windows")

        intervals = defaultdict(list)
        for piece, _, interval, _ in catalog:
            scope = piece["scope"]
            if scope not in scopes or piece["kind"] != "device":
                continue
            lane = piece["resource"]
            resource = graph["resources"][lane]
            stream = int(resource["identity"].removeprefix("stream/"))
            start, finish = (graph["evidence"]["events"][piece[side]]["order"] for side in ("start", "finish"))
            parts = [tasks[stream, a, b] for a, b in facts["resources"][lane]["payload"] if start <= a and b <= finish]
            intervals[scope].append((stream, start, interval, parts))

        # A stream-local sampling window must identify every device interval and its complete payload.
        partitions = {}
        for scope, parts in intervals.items():
            if len({part[0] for part in parts}) == 1 and all(part[3] for part in parts):
                partitions[scope] = [(index, tasks) for _, _, index, tasks in sorted(parts)]
        return partitions

    @classmethod
    def measured(cls, profile, trace):
        counter = HostCounter(profile)
        rows = sdk_rows(trace)
        markers = {row["name"]: Interval.read(row, counter) for row in rows
                   if row["name"].startswith("prologue.sample.")}
        if markers.keys() != {"prologue.sample.begin", "prologue.sample.call", "prologue.sample.end"}:
            return {"reason": "missing_scope_marker"}
        enqueues = AscendTrace.queue_index([Interval.read(row, counter) for row in rows if row.get("cat") == "enqueue"])
        dequeues = ThreadIntervals([Interval.read(row, counter) for row in rows if row.get("cat") == "dequeue"])
        boundary, owned = defaultdict(set), set()
        for row in rows:
            info = row.get("args", {})
            if "Thread Id" not in info or "connection_id" not in info:
                continue
            api = Interval.read(row, counter)
            dequeue = dequeues.enclosing(api)
            origin = enqueues[dequeue.record["args"]["correlation_id"]] if dequeue else api
            for name, marker in markers.items():
                if marker.contains(origin):
                    if name == "prologue.sample.call":
                        owned.add(info["connection_id"])
                    elif info["id"] == "aclrtRecordEvent":
                        boundary[name].add(info["connection_id"])

        flips = task_flips(profile)
        tasks = [r for r in rows if "Task Type" in r.get("args", {}) and not queue_marker(r["args"], flips)]
        endpoints = []
        for name in ("prologue.sample.begin", "prologue.sample.end"):
            endpoints.append([r for r in tasks if r["args"]["connection_id"] in boundary[name]
                              and r["args"]["Task Type"] == "EVENT_RECORD"])
        if any(len(items) != 1 for items in endpoints):
            return {"reason": "unresolved_device_timer"}
        begin, end = (items[0] for items in endpoints)
        stream = begin["args"]["Physic Stream Id"]
        def position(row):
            info = row["args"]
            return Decimal(row["ts"]), info["Batch Id"], info["Task Id"], info["Subtask Id"]

        lane = sorted((r for r in tasks if r["args"]["Physic Stream Id"] == stream), key=position)
        if end not in lane or lane.index(end) <= lane.index(begin):
            return {"reason": "unordered_device_timers"}
        measured = lane[lane.index(begin) + 1:lane.index(end)]
        linked = sum(r["args"]["connection_id"] in owned for r in tasks)
        if len(measured) != linked or any(r["args"]["connection_id"] not in owned for r in measured):
            return {"reason": "scope_and_stream_disagree"}
        return {"tasks": [[*cls.identity(r), round(Decimal(str(r["dur"])) * 1000)] for r in measured]}

    @classmethod
    def export_calls(cls, directory, rank, calls):
        for call in calls:
            if "profile" not in call:
                continue
            root = directory / call["profile"]
            profile, = (root / "native" / f"rank-{rank}").glob("*_ascend_pt")
            with TemporaryDirectory(dir=root) as temporary:
                data = Path(temporary) / "sdk"
                AscendExport.run(root, rank, data)
                call.update(cls.measured(profile, data / "trace.json"))

    @classmethod
    def publish(cls, directory):
        manifest = json.loads((directory / "capture.json").read_text())
        for instance in manifest["instances"]:
            for actor, relative in instance["attention_samples"].items():
                path = directory / relative
                record = json.loads(path.read_text())
                cls.export_calls(directory, int(actor), record["calls"])
                write_json(path, record)
