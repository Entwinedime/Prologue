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
        self.publications = worker["publications"]
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
            owners[arguments["connection_id"]].add((identity, origin.begin, api.begin, api.end))
        return owners

    def document(self):
        scopes = self.markers()
        owners = self.submissions(scopes)
        children = defaultdict(list)
        for call in self.calls:
            if call["parent"] is not None:
                children[call["parent"]].append(scopes[call["id"]].end)

        pending = []
        for row in self.rows:
            details = row.get("args", {})
            if "Physic Stream Id" not in details or details["Task Type"] in self.profiler_controls:
                continue
            pending.append(Span.read(row))

        work, order, lanes = [], [], {}
        physical_predecessor, owned_predecessor = {}, {}
        unbound = defaultdict(Counter)
        for span in sorted(pending, key=lambda entry: (entry.begin, entry.end)):
            details = span.row["args"]
            stream = int(details["Physic Stream Id"])
            previous_end, previous_work = physical_predecessor.get(stream, (None, None))
            physical_predecessor[stream] = (span.end, None)
            possible = owners.get(details.get("connection_id"), ())
            if len(possible) != 1:
                reason = "multiple_submissions" if possible else "missing_submission"
                unbound[reason][details["Task Type"]] += 1
                continue
            (owner, submission, api_begin, api_end), = possible
            if owner is None:
                unbound["outside_calls"][details["Task Type"]] += 1
                continue

            lane = lanes.setdefault(stream, len(lanes))
            index = len(work)
            work.append({
                "id": index, "owner": owner, "slot": bisect_right(children[owner], submission), "lane": lane,
                "kind": details["Task Type"], "name": span.row["name"],
                "submitted_ns": submission, "begin_ns": span.begin, "end_ns": span.end,
                "api_begin_ns": api_begin, "api_end_ns": api_end,
                "previous_device_end_ns": previous_end, "previous_work": previous_work,
            })
            if lane in owned_predecessor:
                order.append([owned_predecessor[lane], index])
            owned_predecessor[lane] = index
            physical_predecessor[stream] = (span.end, index)

        if not work:
            raise ValueError("the trace contains no unambiguously owned device work")
        return {
            "domain": {key: self.manifest[key] for key in ("framework_revision", "model", "configuration")},
            "calls": self.calls, "work": work, "stream_order": order,
            "profile_scope_begins_ns": [scopes[call["id"]].begin for call in self.calls],
            "profile_scope_ends_ns": [scopes[call["id"]].end for call in self.calls],
            "evidence": {"clock": "ascend_profiler_ns", "unbound_device_records": dict(unbound)},
        }

    @staticmethod
    def hardware(directory):
        device_file, = (path for path in directory.rglob("info.json.*") if path.suffix[1:].isdigit())
        profile_file, = directory.rglob("profiler_info_*.json")
        system = json.loads(device_file.read_text(encoding="utf-8"))
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
        device, = system["DeviceInfo"]
        return {
            "platform": "ascend", "driver": system["drvVersion"], "host_cpu": system["CPU"][0]["Type"],
            "cann": profile["cann_version"], "torch_npu": profile["torch_npu_version"],
            "device": {key: device[key] for key in ("ai_core_num", "aiv_num", "aic_frequency", "aiv_frequency")},
        }


class AscendRecording:
    def __init__(self, directory, *, modes=("detailed",)):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.mode = self.manifest["mode"]
        if self.mode not in modes:
            raise ValueError(f"observation mode {self.mode} does not qualify for this use")
        self.windows = self.manifest.get("windows")
        self.profiles = {}
        for window in self.windows or ():
            for identity, trace in window["traces"].items():
                rank = int(identity)
                if rank in self.profiles:
                    raise ValueError(f"the recording repeats the profile of rank {rank}")
                worker = directory / window["directory"] / f"rank-{rank}" / "worker.json"
                self.profiles[rank] = (worker, directory / trace)

    @property
    def delivery(self):
        return self.manifest["delivery"] if self.windows is None else self.windows[0]["delivery"]

    @property
    def domain(self):
        return {key: self.manifest[key] for key in ("framework_revision", "model", "configuration")}

    def bindings(self):
        ranks = range(self.manifest["configuration"]["tp"])
        if self.manifest["placement"] == "single_device":
            workers = self.manifest["workers"]
            bindings = [workers[str(rank)]["representative"] for rank in ranks]
        else:
            bindings = list(ranks)
        if set(bindings) != self.profiles.keys():
            raise ValueError("observed ranks do not match the recording's rank bindings")
        return bindings

    def publication_rank(self):
        rank, = (int(rank) for rank, worker in self.manifest["workers"].items() if worker["publication_owner"])
        return rank

    def profile_directory(self, rank):
        return self.profiles[rank][1].parents[1]

    def read(self, rank=None):
        if rank is not None:
            worker, trace = self.profiles[rank]
        elif self.windows is None:
            worker = self.directory / "worker.json"
            trace = self.directory / self.manifest["trace"]
        else:
            raise ValueError("the recording has multiple rank profiles")

        records = json.loads(worker.read_text(encoding="utf-8"))
        rows = json.loads(trace.read_text(encoding="utf-8"), parse_float=Decimal)
        return AscendEvidence(self.manifest, records, rows)
