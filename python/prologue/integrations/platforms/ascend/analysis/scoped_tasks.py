"""Match the task sequence from a bounded SDK observation to owned source-graph work."""

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import write_json
from prologue.integrations.platforms.ascend.analysis.clock import HostCounter
from prologue.integrations.platforms.ascend.analysis.commands import StreamCommands
from prologue.integrations.platforms.ascend.analysis.sdk_rows import queue_marker, task_flips
from prologue.integrations.platforms.ascend.analysis.trace import AscendTrace, Interval, ThreadIntervals
from prologue.integrations.platforms.ascend.runtime.export import AscendExport


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
        rows = [row for row in json.loads(trace.read_text(), parse_float=Decimal) if row.get("ph") == "X"]
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
