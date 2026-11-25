"""Read the native recording and index its graph coordinates for independent comparison."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property

from prologue.integrations.platforms.ascend.trace_store import (
    EVENT_WAITS,
    STREAM_WAITS,
    Interval,
    RawTraceIndex,
    ThreadIntervals,
    device_streams,
    event_records,
    sdk_rows,
    task_flips,
)


@dataclass(frozen=True, slots=True)
class OmittedRecord:
    thread: int
    order: int
    posted_ns: int
    completed_device_ns: int


@dataclass(frozen=True, slots=True)
class DeviceRelease:
    submitted: int
    previous: int | None
    preceding_window_ns: tuple[int | None, int] | None
    first_command: bool
    submitted_at_completion: bool
    completion: int | None
    omitted: tuple[OmittedRecord, ...]


class SdkStreams:
    fields = ("connection_id", "Batch Id", "Task Id", "Subtask Id")
    boundaries = frozenset({"EVENT_RECORD", "EVENT_WAIT", "PROFILING_ENABLE", "PROFILING_DISABLE"})

    def __init__(self, rows, clock, flips):
        self.clock = clock
        self.records = {}
        self.connections = defaultdict(list)
        self.counts = Counter()
        self.logical_gaps = {}
        self.logical_slots = {}
        self.observer_payload = set()
        for stream, tasks in device_streams(rows, clock, flips).items():
            self.counts[stream] = len(tasks)
            for order, row in enumerate(tasks):
                identity = stream, order
                self.records[identity] = row
                self.connections[row["args"]["connection_id"]].append(identity)

    @classmethod
    def key(cls, args):
        return tuple(args[name] for name in cls.fields)

    def event(self, connection):
        identity, = self.connections[connection]
        if self.records[identity]["args"]["Task Type"] != "EVENT_RECORD":
            raise ValueError("native event has a different SDK task kind")
        return identity

    def span(self, identity):
        stream, occurrence = identity
        position = occurrence + bisect_right(self.logical_gaps.get(stream, ()), occurrence)
        return 2 * position, 2 * position + 1

    def task(self, stream, order):
        slot = order // 2
        logical = self.logical_slots.get(stream, ())
        index = bisect_left(logical, slot)
        if index < len(logical) and logical[index] == slot:
            return None
        return self.records[stream, slot - index]

    def omitted_records(self, stream, ordinal, coordinates, queues, observer_joins):
        records = []
        resource = f"stream/{stream}"

        # Elided waits add logical slots. Physical order still follows consecutive SDK records.
        for previous in range(ordinal - 1, -1, -1):
            identity = stream, previous
            _, end = self.span(identity)
            retained = coordinates.get((resource, end))
            if retained is not None and retained not in observer_joins:
                return retained, tuple(reversed(records))

            # Excluded observation work can still occupy the physical stream before modeled work.
            task = self.records[identity]
            args = task["args"]
            if (args["Task Type"] != "EVENT_RECORD" and identity not in self.observer_payload
                    or args["connection_id"] not in queues.apis):
                break
            completed = self.clock.interval(task)[1]
            if not self.clock.begin_ns <= completed <= self.clock.end_ns:
                break
            try:
                post = queues.release(args)
            except ValueError:
                break
            records.append(OmittedRecord(post.thread, post.begin, post.record["begin_raw_ns"], completed))
        return None, ()

    def release_evidence(self, graph, launches, queues, calibration, observer_joins):
        coordinates = {
            (graph["resources"][event["resource"]]["identity"], point["order"]): index
            for index, (event, point) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True))
        }
        releases = {}
        for identity, (post, order) in launches.items():
            stream, ordinal = identity
            resource = f"stream/{stream}"
            recorded = coordinates[resource, order]
            target, completion = recorded, None
            kind = self.records[identity]["args"]["Task Type"]
            if kind in {"EVENT_RECORD", "EVENT_WAIT"}:
                begin, end = self.span(identity)

                # A signal can keep only its completion, or retain both physical command boundaries.
                target = coordinates.get((resource, begin), recorded)
                returned = coordinates.get((resource, end))
                if returned != target:
                    completion = returned
            submitted = coordinates[f"thread/{post.thread}", post.begin]
            preceding = stream, ordinal - 1
            prior = self.records.get(preceding)
            previous, seen, omitted = None, None, ()
            if prior is not None:
                previous = coordinates.get((f"stream/{stream}", self.span(preceding)[1]))
                completed = self.clock.interval(prior)[1]
                if completed < self.clock.begin_ns:

                    # An earlier device tick finished by the first snapshot's RAW return.
                    seen = None, self.clock.samples[0]["end_raw_ns"]
                elif completed <= self.clock.end_ns:
                    seen = self.clock.bounds(completed)
                    entered = post.record["begin_raw_ns"]
                    if seen[1] > entered:

                        # Retained causal edges tighten the same measured clock range.
                        seen = calibration.point_window(calibration.point(completed, True))
                        if seen[1] > entered and (previous is None or previous in observer_joins):
                            previous, omitted = self.omitted_records(
                                stream, ordinal, coordinates, queues, observer_joins
                            )
            releases[target] = DeviceRelease(submitted, previous, seen, ordinal == 0, recorded == completion,
                                             completion, omitted)
        return releases

    def inspect_coordinates(self, facts, apis, commands):
        logical = defaultdict(dict)
        for effect in facts["native_effects"]:
            for command in effect["commands"]:
                if command["sdk"] is not None:
                    continue
                begin, end = command["begin"], command["end"]
                stream = int(facts["resources"][effect["resource"]]["identity"].split("/")[1])
                if begin % 2 or end != begin + 1 or begin // 2 in logical[stream]:
                    raise ValueError("elided native waits repeat or change a stream command slot")
                logical[stream][begin // 2] = apis[effect["connection"]]

        positions = {stream: sorted(calls) for stream, calls in logical.items()}
        for stream, slots in positions.items():
            before, after = [], []
            for (peer, order), row in self.records.items():
                if peer != stream:
                    continue
                api = apis.get(row["args"]["connection_id"])
                if api is not None:
                    before.append((api.end, order + 1))
                    after.append((api.begin, order))
            before.sort()
            after.sort()
            lower, upper = [0], [self.counts[stream]] * (len(after) + 1)
            for _, order in before:
                lower.append(max(lower[-1], order))
            for index in range(len(after) - 1, -1, -1):
                upper[index] = min(upper[index + 1], after[index][1])
            ends, starts = [time for time, _ in before], [time for time, _ in after]

            # A zero-submission wait occupies a slot only at the unique gap proved by native API ordering.
            for index, slot in enumerate(slots):
                api = logical[stream][slot]
                left = lower[bisect_right(ends, api.begin)]
                right = upper[bisect_left(starts, api.end)]
                if slot - index != left or left != right:
                    raise ValueError("elided native wait changes or lacks its unique raw SDK position")

        self.logical_gaps = {
            stream: [slot - index for index, slot in enumerate(slots)] for stream, slots in positions.items()
        }
        self.logical_slots = positions
        for identity, span in commands.items():
            if span != self.span(identity):
                raise ValueError("native command changes its actual SDK occurrence coordinate")

        checked = 0
        for point in facts["measurements"]["coordinates"]:
            resource = facts["resources"][point["resource"]]
            if resource["kind"] != "stream":
                continue
            stream, order = int(resource["identity"].split("/")[1]), point["order"]
            task = self.task(stream, order)
            measured = None if task is None else self.clock.interval(task)[order % 2]
            if point["time_ns"] != measured:
                raise ValueError("device boundary changes or loses its original SDK measurement")
            checked += measured is not None
        return checked


class SdkRecording(RawTraceIndex):
    waits = STREAM_WAITS

    def __init__(self, capture, rank, native, facts, sdk):
        self.native, self.facts = native, facts
        self.logical_coordinates = {}
        self.observer_joins = set()
        profile, = (capture / "native" / f"rank-{rank}").glob("*_ascend_pt")
        rows = sdk_rows(sdk)
        flips = task_flips(profile)
        super().__init__(profile, capture / "runtime", native, rows, flips)
        self.sdk = SdkStreams(rows, self.clock, flips)
        self.runtime_records = {(interval.thread, interval.begin): interval.record for interval in self.runtime}
        self.require(len(self.runtime_records) == len(self.runtime), "raw runtime repeats an API interval identity")
        self.source_scopes = {origin: index for index, scope in enumerate(facts["scopes"])
                              for origin in scope["bindings"]["source_scopes"]}
        self.queries = {query["scope"]: {**query, "thread": native["scopes"][query["scope"]]["thread"]}
                        for query in native["state_queries"]}
        self.queried_points = {row["point"] for row in self.queries.values() if row["kind"] == "event"}
        for query in self.queries.values():
            self.native_interval(query)
        self.index_graph(facts)

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

    def owner(self, api):
        origin = self.origin(api)
        scope = self.marker_ranges.enclosing(origin)
        if scope is None or self.excluded_ranges.enclosing(origin) is not None:
            return None
        return int(scope.record["name"].split("/")[1])

    def point_connection(self, point):
        return self.point_api(point).record["args"]["connection_id"]

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
        work, waits = {}, defaultdict(list)
        for interval, producer in event_records(self.runtime, EVENT_WAITS):
            if interval.end <= self.window.begin or self.window.end <= interval.begin:
                continue
            api = self.native_api(interval)
            owner = self.owner(api)
            if owner is None:
                continue
            self.require(producer is not None, "native event consumer has no recorded generation")
            recorded = self.native_api(producer)
            connection = recorded.record["args"]["connection_id"]
            waits[owner, connection].append((interval.record, api))
            producer = self.owner(recorded)
            if producer in peers[owner]:
                self.require(
                    work.setdefault(producer, connection) == connection,
                    "native Work has competing completion generations",
                )
        return work, waits
