"""Associate device tasks with the native submitted scopes in one Ascend trace."""

import json
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Span:
    row: dict
    begin: int
    end: int
    thread: int

    @classmethod
    def read(cls, row):
        begin = Decimal(row["ts"]) * 1000
        end = begin + Decimal(row["dur"]) * 1000
        return cls(row, round(begin), round(end), int(row["tid"]))

    def contains(self, child):
        return self.begin <= child.begin and child.end <= self.end


class HostIntervals:
    def __init__(self, spans):
        grouped = defaultdict(list)
        for span in spans:
            grouped[span.thread].append(span)

        self.threads = {}
        for thread, lane in grouped.items():
            lane.sort(key=lambda span: (span.begin, -span.end))
            starts, parents, stack = [], [], []
            for index, span in enumerate(lane):
                while stack and not lane[stack[-1]].contains(span):
                    stack.pop()
                starts.append(span.begin)
                parents.append(stack[-1] if stack else -1)
                stack.append(index)
            self.threads[thread] = starts, parents, lane

    def enclosing(self, span):
        starts, parents, lane = self.threads.get(span.thread, ((), (), ()))
        index = bisect_right(starts, span.begin) - 1
        while index >= 0:
            if lane[index].contains(span):
                return lane[index]
            index = parents[index]
        return None


class AscendEvidence:
    marker_prefix = "prologue.scope."
    profiler_controls = {"PROFILING_ENABLE", "PROFILING_DISABLE"}

    def __init__(self, manifest, worker, rows):
        self.manifest = manifest
        self.calls = worker["calls"]
        self.rows = [row for row in rows if row.get("ph") == "X"]

    def markers(self):
        scopes = {}
        for row in self.rows:
            if not row["name"].startswith(self.marker_prefix):
                continue
            identity = int(row["name"].removeprefix(self.marker_prefix))
            if identity in scopes:
                raise ValueError(f"duplicate trace marker for call {identity}")
            scopes[identity] = Span.read(row)
        if scopes.keys() != {call["id"] for call in self.calls}:
            raise ValueError("the device profile does not cover the observed call window")
        return scopes

    def submissions(self, scopes):
        enqueues = {}
        dequeues = []
        for row in self.rows:
            category = row.get("cat")
            if category == "enqueue":
                enqueues[row["args"]["correlation_id"]] = Span.read(row)
            elif category == "dequeue":
                dequeues.append(Span.read(row))
        queue = HostIntervals(dequeues)
        markers = HostIntervals(scopes.values())

        # The SDK connects device work to the API. A queued API keeps the originating thread's scope.
        owners = defaultdict(set)
        for row in self.rows:
            arguments = row.get("args", {})
            if "connection_id" not in arguments or "Thread Id" not in arguments:
                continue
            api = Span.read(row)
            dequeue = queue.enclosing(api)
            origin = enqueues.get(dequeue.row["args"]["correlation_id"]) if dequeue else api
            if origin is None:
                continue
            marker = markers.enclosing(origin)
            identity = int(marker.row["name"].removeprefix(self.marker_prefix)) if marker else None
            owners[arguments["connection_id"]].add((identity, origin.begin))
        return owners

    def document(self):
        scopes = self.markers()
        owners = self.submissions(scopes)
        children = defaultdict(list)
        for call in self.calls:
            if call["parent"] is not None:
                children[call["parent"]].append(scopes[call["id"]].end)

        pending, unbound = [], defaultdict(Counter)
        for row in self.rows:
            details = row.get("args", {})
            if "Physic Stream Id" not in details or details["Task Type"] in self.profiler_controls:
                continue
            possible = owners.get(details.get("connection_id"), ())
            if len(possible) != 1:
                reason = "multiple_submissions" if possible else "missing_submission"
                unbound[reason][details["Task Type"]] += 1
                continue
            (owner, submission), = possible
            if owner is None:
                unbound["outside_calls"][details["Task Type"]] += 1
                continue
            pending.append((Span.read(row), submission, owner))

        work, order, lanes, predecessor = [], [], {}, {}
        for span, submission, owner in sorted(pending, key=lambda entry: (entry[0].begin, entry[0].end)):
            details = span.row["args"]
            stream = int(details["Physic Stream Id"])
            lane = lanes.setdefault(stream, len(lanes))
            index = len(work)
            work.append({
                "id": index, "owner": owner, "slot": bisect_right(children[owner], submission), "lane": lane,
                "kind": details["Task Type"], "name": span.row["name"],
                "submitted_ns": submission, "begin_ns": span.begin, "end_ns": span.end,
            })
            if lane in predecessor:
                order.append([predecessor[lane], index])
            predecessor[lane] = index

        if not work:
            raise ValueError("the trace contains no unambiguously owned device work")
        return {
            "domain": {key: self.manifest[key] for key in ("framework_revision", "model", "configuration")},
            "calls": self.calls, "work": work, "stream_order": order,
            "evidence": {"clock": "ascend_profiler_ns", "unbound_device_records": dict(unbound)},
        }

    @classmethod
    def read(cls, directory):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["mode"] != "detailed":
            raise ValueError("template compilation requires a detailed native observation")
        worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        trace = json.loads((directory / manifest["trace"]).read_text(encoding="utf-8"), parse_float=Decimal)
        return cls(manifest, worker, trace).document()
