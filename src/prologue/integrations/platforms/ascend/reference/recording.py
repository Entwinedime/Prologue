"""Index raw evidence and comparison coordinates for an independent native reference."""

import json
from collections import Counter, defaultdict
from decimal import Decimal
from functools import cached_property

from prologue.integrations.platforms.ascend.analysis.clock import AscendClock, HostCounter
from prologue.integrations.platforms.ascend.analysis.sdk_rows import Interval, ThreadIntervals, task_flips

from .queue_evidence import QueueEvidence
from .reference_streams import SdkStreams


class SdkRecording:
    waits = {"aclrtStreamWaitEvent", "aclrtStreamWaitEventWithTimeout"}

    def __init__(self, capture, rank, native, facts, sdk):
        self.native, self.facts = native, facts
        self.logical_coordinates = {}
        self.observer_joins = set()
        profile, = (capture / "native" / f"rank-{rank}").glob("*_ascend_pt")
        counter = HostCounter(profile)
        device, = profile.glob("PROF_*/device_*")
        self.clock = AscendClock(device, native["device_clock"])
        flips = task_flips(profile)
        rows = json.loads(sdk.read_text(), parse_float=Decimal)
        self.sdk = SdkStreams(rows, self.clock, flips)
        self.index_sdk(rows, counter)
        self.index_markers(native)
        self.index_runtime(capture, native, facts, rows, flips)
        self.index_graph(facts)

    def index_sdk(self, rows, counter):
        self.apis = {}
        self.enqueues = {}
        self.dequeue_ids = {}
        self.host_measurements = {}
        dequeues = []
        for row in rows:
            if row.get("ph") != "X":
                continue
            args = row.get("args", {})
            if "Task Type" not in args and (row.get("cat") in {"enqueue", "dequeue"} or "connection_id" in args):
                interval = Interval.read(row, counter)

                # Reconstruct expected times from raw SDK coordinates, never from interpreted measurements.
                for position in (interval.begin, interval.end):
                    self.host_measurements[f"thread/{interval.thread}", position] = counter.framework_raw + round(
                        Decimal(position - counter.anchor) * 1_000_000_000 / counter.frequency
                    )
            if "Task Type" in args:
                continue
            elif row.get("cat") in {"enqueue", "dequeue"}:
                if row["cat"] == "enqueue":
                    self.enqueues[args["correlation_id"]] = interval
                else:
                    dequeues.append(interval)
                    self.dequeue_ids[args["correlation_id"]] = interval
            elif "Thread Id" in args and "connection_id" in args:
                self.require(args["connection_id"] not in self.apis, "raw SDK repeats an API connection")
                self.apis[args["connection_id"]] = interval

        self.api_ranges = ThreadIntervals(list(self.apis.values()))
        self.dequeues = ThreadIntervals(dequeues)
        self.enqueue_ranges = ThreadIntervals(list(self.enqueues.values()))

    def index_markers(self, native):
        scopes, excluded, windows = [], [], []
        self.host_records = {}
        self.points = {}
        signals = {index for index, point in enumerate(native["points"]) if point["side"] == "signal"}
        for row in native["host_observation"]["intervals"]:
            interval = self.interval(row)
            name = row["name"]
            if name == "prologue.window":
                windows.append(interval)
            elif name.startswith("prologue.scope/"):
                scopes.append(interval)
                identity = int(name.split("/")[1])
                self.require(identity not in self.host_records, "raw recording repeats a host scope")
                self.host_records[identity] = row
            elif name.startswith("prologue.point/"):
                identity = int(name.split("/")[1])
                self.require(identity not in self.points, "raw recording repeats a device point")
                self.points[identity] = interval
                if identity not in signals:
                    excluded.append(interval)
            elif name in {"prologue.observer", "prologue.auxiliary"}:
                excluded.append(interval)
        self.window, = windows
        for scope in native["scopes"]:
            if "service" in scope:
                self.require(scope["id"] not in self.host_records, "native service has competing host records")
                self.host_records[scope["id"]] = {**scope["host_interval"], "thread": scope["thread"]}
        self.require(
            set(self.host_records) == set(range(1, len(native["scopes"]))), "raw host scope observations are incomplete"
        )
        self.scopes, self.excluded = ThreadIntervals(scopes), ThreadIntervals(excluded)

    def index_runtime(self, capture, native, facts, rows, flips):
        process = native["host_observation"]["process"]
        self.runtime = [
            json.loads(line) for line in (capture / "runtime" / f"{process}.jsonl").read_text().splitlines()
        ]
        self.runtime_records = {(row["thread"], row["begin_counter"]): row for row in self.runtime}
        self.require(len(self.runtime_records) == len(self.runtime), "raw runtime repeats an API interval identity")
        self.source_scopes = {origin: index for index, scope in enumerate(facts["scopes"])
                              for origin in scope["bindings"]["source_scopes"]}
        self.closed_scopes = set()
        for scope in native["scopes"]:
            if "service" in scope or scope["parent"] in self.closed_scopes:
                self.closed_scopes.add(scope["id"])

        # Decode driver postings directly from the recording, before checking any projected edges.
        calls = (self.interval(row) for row in self.runtime if row["api"].startswith("halSq"))
        self.queues = QueueEvidence(calls, self.apis, rows, flips)

        # Actual RAW samples override SDK reconstruction at their own coordinates, including service threads.
        services = [{**scope["host_interval"], "thread": scope["thread"]}
                    for scope in native["scopes"] if "service" in scope]
        self.queries = {query["scope"]: {**query, "thread": native["scopes"][query["scope"]]["thread"]}
                        for query in native["state_queries"]}
        self.queried_points = {row["point"] for row in self.queries.values() if row["kind"] == "event"}
        for row in [*native["host_observation"]["intervals"], *services, *self.runtime, *self.queries.values()]:
            for side in ("begin", "end"):
                for resource in (f"thread/{row['thread']}", f"service/{row['thread']}"):
                    self.host_measurements[resource, row[f"{side}_counter"]] = row[f"{side}_raw_ns"]

    def index_graph(self, facts):

        # These immutable source intervals serve every API check; copies must not rescan the whole recording.
        self.spans = Counter((span["scope"], span["resource"], span["begin"], span["end"], span["kind"])
                             for span in facts["spans"])
        self.span_ranges = ThreadIntervals(
            [Interval(begin, end, resource, {}) for _, resource, begin, end, _ in self.spans]
        )
        self.wait_intervals = {(resource, begin, end) for _, resource, begin, end, kind in self.spans if kind == "wait"}
        self.payload_ranges = ThreadIntervals([
            Interval(begin, end, index, {})
            for index, resource in enumerate(facts["resources"]) for begin, end in resource["payload"]
        ])
        self.transfer_signals = {
            (
                tuple(sorted(edge["target"].items())),
                facts["spans"][edge["source"]["span"]]["resource"],
                edge["source"]["order"],
            )
            for edge in facts["dependencies"]
            if edge["kind"] == "signal"
        }

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def interval(row):
        return Interval(row["begin_counter"], row["end_counter"], row["thread"], row)

    def origin(self, api):
        dequeue = self.dequeues.enclosing(api)
        return self.enqueues[dequeue.record["args"]["correlation_id"]] if dequeue else api

    def owner(self, api):
        origin = self.origin(api)
        scope = self.scopes.enclosing(origin)
        if scope is None or self.excluded.enclosing(origin) is not None:
            return None
        return int(scope.record["name"].split("/")[1])

    def native_api(self, interval):
        matches = [
            api for api in self.api_ranges.inside(interval) if api.record["args"]["id"] == interval.record["api"]
        ]
        self.require(len(matches) == 1, "raw stream operation lacks one SDK API")
        return matches[0]

    def point_connection(self, point):
        enqueue, = self.enqueue_ranges.inside(self.points[point])
        dequeue = self.dequeue_ids[enqueue.record["args"]["correlation_id"]]
        api, = [api for api in self.api_ranges.inside(dequeue) if api.record["args"]["id"] == "aclrtRecordEvent"]
        return api.record["args"]["connection_id"]

    def check_device_end(self, graph, event, connection):
        identity = self.sdk.event(connection)
        task = self.sdk.records[identity]
        resource = graph["resources"][graph["events"][event]["resource"]]
        self.require((resource["kind"], resource["identity"]) == ("stream", f"stream/{identity[0]}"),
                     "native signal changes its stream")
        measured = self.clock.interval(task)[1]
        self.require(
            graph["evidence"]["events"][event]["time_ns"] == measured,
            "native signal changes its SDK completion measurement",
        )

    @cached_property
    def event_generations(self):
        peers = {
            scope["id"]: {port["producer"] for port in scope["ports"]
                          if port["use"] == "signal" and port["direction"] == "input" and "producer" in port}
            for scope in self.native["scopes"]
        }
        records, work, waits = {}, {}, defaultdict(list)
        for row in sorted(self.runtime, key=lambda row: row["begin_counter"]):
            if row["api"] == "aclrtRecordEvent":
                records[row["event"]] = row
            elif row["api"] == "aclrtDestroyEvent":
                records.pop(row["event"], None)
            elif row["api"] in self.waits | {"aclrtSynchronizeEvent", "aclrtSynchronizeEventWithTimeout"}:
                interval = self.interval(row)
                if interval.end <= self.window.begin or self.window.end <= interval.begin:
                    continue
                api = self.native_api(interval)
                owner = self.owner(api)
                if owner is None:
                    continue
                producer = records.get(row["event"])
                self.require(producer is not None, "native event consumer has no recorded generation")
                recorded = self.native_api(self.interval(producer))
                connection = recorded.record["args"]["connection_id"]
                waits[owner, connection].append((row, api))
                producer = self.owner(recorded)
                if producer in peers[owner]:
                    self.require(
                        work.setdefault(producer, connection) == connection,
                        "native Work has competing completion generations",
                    )
        return work, waits
