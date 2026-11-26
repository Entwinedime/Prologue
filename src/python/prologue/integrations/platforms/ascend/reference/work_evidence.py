"""Qualify the work intervals and fences that survive projection against original native time."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property

from prologue.integrations.platforms.ascend.reference.recording import SdkRecording
from prologue.integrations.platforms.ascend.trace_store import (
    HOST_WAITS,
    Interval,
    native_interval,
)


@dataclass(frozen=True, slots=True)
class HostReplay:
    """Host intervals in unscaled RAW nanoseconds for same-run service replay."""

    observer: list[tuple[int, int, int]]
    dequeues: list[tuple[int, int, int]]
    waits: list[tuple[int, int, int]]
    calls: list[Interval]


class WorkEvidence(SdkRecording):
    """Check that native host and device intervals survive graph projection with their original owners and
    boundaries.
    """

    @staticmethod
    def peer_exchange(scope):
        participants = scope.get("communication")
        return participants is not None and (
            "peer_rows" in scope or participants["required"] == participants["members"]
        )

    def inspect_host_fences(self, graph):
        resources = graph["resources"]
        locations = {
            (node["resource"], point["order"]): event
            for event, (node, point) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True))
            if resources[node["resource"]]["kind"] == "thread"
        }
        fences = {}

        for scope, lane, begin, end, kind in self.spans:
            resource = resources[lane]
            if kind != "wait" or resource["kind"] != "thread":
                continue

            source, = self.facts["scopes"][scope]["bindings"]["source_scopes"]
            raw = self.native["scopes"][source]
            if "service" in raw:
                interval = self.scopes[source].record
                if (begin, end) == (interval["begin_counter"], interval["end_counter"]):
                    continue

            tid = int(resource["identity"].split("/")[1])
            call = self.runtime_records.get((tid, begin))
            self.require(call is not None and call["api"] in HOST_WAITS
                         and call["end_counter"] == end and call["result"] == 0,
                         "recorded host fence must use its native runtime entry and return")
            api = self.native_api(native_interval(call))
            self.require(self.owner(api) == source, "host fence changes its native owner")
            fences[locations[lane, begin], locations[lane, end]] = call

        for scope in graph["scopes"]:
            for port in scope["ports"]:
                if "wait" not in port:
                    continue
                wait = port["wait"]
                call = fences[wait["begin"], port["event"]]

                if call["api"] != "aclrtSynchronizeDeviceWithTimeout":
                    stream = resources[wait["resource"]]
                    self.require(call["stream_status"] == 0 and stream["identity"] == f"stream/{call['stream']}",
                                 "host fence changes its recorded execution stream")

        return frozenset(fences)

    def replay_host_intervals(self):
        observer = [self.host_interval(interval) for interval in self.excluded]
        dequeues = []
        for scope, read in self.queries.items():
            wrapper = self.scopes[scope].record
            observer.append((wrapper["thread"], wrapper["begin_raw_ns"], read["begin_raw_ns"]))
            observer.append((wrapper["thread"], read["end_raw_ns"], wrapper["end_raw_ns"]))

        # Queue entries owned by the observer can use a different native thread.
        for interval, observer_owned in self.dequeued_intervals():
            dequeues.append(interval)
            if observer_owned:
                observer.append(interval)

        repeated = set(map(tuple, self.facts["measurements"].get("repeated_host_waits", ())))
        waits = list(self.reserved_host_intervals(repeated))
        for lane, begin, end in self.wait_intervals:
            resource = self.facts["resources"][lane]
            if resource["kind"] == "thread":
                identity = resource["identity"]
                thread = int(identity.split("/")[1])
                waits.append((thread, self.host_times[identity, begin], self.host_times[identity, end]))

        # Runtime indexing uses device-visible counter coordinates. Replay compares kernel events in RAW time.
        calls = []
        for interval in self.runtime:
            row = interval.record
            calls.append(Interval(row["begin_raw_ns"], row["end_raw_ns"], interval.thread, row))

        return HostReplay(observer, dequeues, waits, calls)

    def inspect_measurements(self, external):
        checked = 0
        unmeasured = set()
        observed = defaultdict(set)

        for point in self.facts["measurements"]["coordinates"]:
            resource = self.facts["resources"][point["resource"]]
            lane = resource["identity"], resource["role"], resource["serial"]
            if lane in external:
                self.require(resource["kind"] == "service" and point["time_ns"] is None,
                             "single-device operand staging claims target transport time")
                observed[lane].add(point["order"])
                continue

            if resource["kind"] == "stream":
                if point["time_ns"] is None:
                    unmeasured.add((point["resource"], point["order"]))
                continue

            key = resource["identity"], point["order"]
            self.require(key in self.host_times, f"host boundary has no original time source: {key}")
            self.require(
                point["time_ns"] == self.host_times[key],
                f"host measurement is missing or changes its original time: {key}",
            )
            checked += 1

        # inspect() independently checks these native commands against raw API and SDK task records.
        native_only = {(effect["resource"], command[side])
                       for effect in self.facts["native_effects"] for command in effect["commands"]
                       if command["sdk"] is None for side in ("begin", "end")}
        self.require(observed == external, "external service boundaries differ from original posts and waits")
        self.require(
            unmeasured <= native_only, "device boundary loses a recorded timestamp or lacks an observed native command"
        )
        logical = self.logical_coordinates.keys()

        return {"checked_host_measurements": checked, "external_transport_endpoints": sum(map(len, observed.values())),
                "unmeasured_device_endpoints": len(unmeasured - logical),
                "logical_wait_boundaries": len(unmeasured & logical)}

    def device_intervals(self):
        intervals = {}
        for scope in self.native["scopes"]:
            if "device_entry" not in scope:
                continue
            endpoints = []

            for field, side in (("device_entry", "entry"), ("device_return", "return")):
                point = scope[field]
                self.require(self.native["points"][point] == {"scope": scope["id"], "side": side},
                             "device interval refers to another call's original marker")
                endpoints.append(self.sdk.event(self.point_connection(point)))

            first, last = endpoints
            self.require(first[0] == last[0], "native device interval changes streams")
            begin, end = self.sdk.span(first)[1], self.sdk.span(last)[1]
            self.require(begin <= end, "native device interval reverses its original markers")
            intervals[scope["id"]] = Interval(begin, end, first[0], {})

        return intervals

    def inspect_boundaries(self, graph, device_intervals):
        spans = self.spans
        resources = {
            (resource["kind"], resource["identity"]): index for index, resource in enumerate(graph["resources"])
        }
        self.require(len(resources) == len(graph["resources"]), "coarse graph repeats a native resource")

        # Pure CPU declarations still require independent proof that they own no device payload.
        device_owners = {
            self.owner(api) for connection, api in self.apis.items() if self.sdk.connections.get(connection)
        }
        counts = Counter()

        for index, scope in enumerate(graph["scopes"]):
            origin, = scope["bindings"]["source_scopes"]
            if origin == 0:
                continue

            raw = self.native["scopes"][origin]
            thread = raw["thread"]
            begin = end = self.scopes[origin].record
            self.require(begin["thread"] == thread, f"scope {index} changes its native caller thread")
            ports = {port["name"]: port for port in scope["ports"]}
            kind = "wait" if "service" in raw else "boundary" if raw.get("envelope", False) else "host"

            if "communication" in raw and "device_entry" not in raw:
                kind = "collective" if self.peer_exchange(raw) else "communication"
            host = resources["thread", f"thread/{thread}"]
            if origin in self.queries:
                self.require(spans[index, host, begin["begin_counter"], end["end_counter"], "boundary"] == 1,
                             f"query {index} loses its uncharged caller envelope")
                read = self.queries[origin]
                self.require(
                    begin["begin_counter"] <= read["begin_counter"] <= read["end_counter"] <= end["end_counter"],
                    f"query {index} escapes its original caller interval",
                )
                self.check_boundary(graph, ports["caller.return"]["event"], host, end, "end", index)
                counts["caller_boundaries"] += 1
                begin = end = read
                kind = "host"

            self.require(spans[index, host, begin["begin_counter"], end["end_counter"], kind] == 1,
                         f"scope {index} omits, duplicates or changes its native host interval")
            for name, side, record in (("call.enter", "begin", begin), ("call.return", "end", end)):
                self.check_boundary(graph, ports[name]["event"], host, record, side, index)
                counts["host_boundaries"] += 1

            counts["host_intervals"] += 1
            if raw["id"] in device_intervals:
                interval = device_intervals[raw["id"]]
                if self.inspect_device_interval(graph, resources, raw, ports, interval):
                    counts["device_intervals"] += 1

            if scope["role"] == "segment" and raw["bindings"].get("segment", {}).get("host_only", False):
                self.require(origin not in device_owners, f"CPU-only scope {index} hides native device work")
                counts["cpu_only_regions"] += 1

            if "service" in raw:
                service = resources["service", f"service/{thread}"]
                self.require(
                    not graph["resources"][service]["serial"],
                    "synchronous service is charged as a serial caller resource",
                )
                self.require(
                    spans[index, service, begin["begin_counter"], end["end_counter"], raw["service"]["kind"]] == 1,
                    f"scope {index} omits, duplicates or changes its native service interval",
                )
                self.check_boundary(graph, ports["service.complete"]["event"], service, end, "end", index)
                counts["service_intervals"] += 1

            if "native_transfer" in raw:

                # Each payload keeps its own readiness interval; merging KV and later metadata invents early reads.
                for access in scope["kv_accesses"] + scope["slot_accesses"]:
                    self.check_boundary(graph, access["start"], service, begin, "begin", index)
                    self.check_boundary(graph, access["finish"], service, end, "end", index)

                counts["derived_state_accesses"] += len(scope["kv_accesses"]) + len(scope["slot_accesses"])
            else:
                counts["declared_state_accesses"] += len(raw.get("state_accesses", ()))

        return dict(counts)

    def inspect_device_interval(self, graph, resources, raw, ports, interval):
        """Check that projected device endpoints stay inside their independently observed markers."""

        endpoints = [ports[name]["event"] for name in ("device.enter", "device.complete") if name in ports]
        self.require(len(endpoints) in {0, 2}, "device interval loses one public endpoint")

        if endpoints:
            stream = resources["stream", f"stream/{interval.thread}"]
            self.require(all(graph["events"][event]["resource"] == stream for event in endpoints),
                         "device interval changes its original stream")
            first, last = (graph["evidence"]["events"][event]["order"] for event in endpoints)

            # Semantic trimming may remove observer-only tails, but cannot escape the original markers.
            self.require(interval.begin <= first <= last <= interval.end,
                         "semantic device interval escapes its original marker envelope")
            for event, original, position, point in zip(
                endpoints, (interval.begin, interval.end), (first, last),
                (raw["device_entry"], raw["device_return"]), strict=True,
            ):

                # An unconsumed recorder command only witnesses when its semantic port was already ready.
                if original == position and point not in self.queried_points:
                    self.observer_joins.add(event)

        return bool(endpoints)


    def inspect_services(self):
        services = {scope["id"]: scope for scope in self.native["scopes"] if "service" in scope}
        expected = Counter()
        observed = set()
        queues = {}

        # Reconstruct ownership from original submissions, without reading the producer's task assignments.
        for stream, post in self.queues.submissions:
            row = post.record
            if row["service"] < 0:
                continue

            scope = services[row["service"]]
            matches = [index for index, call in enumerate(scope["service"]["executions"])
                       if call["thread"] == row["thread"] and call["begin_counter"] <= row["begin_counter"]
                       and row["end_counter"] <= call["end_counter"]]
            self.require(len(matches) == 1, "private submission has no unique declared synchronous execution")
            previous = queues.get(stream)
            self.require(previous is None or previous[0] == scope["thread"] and previous[1] <= row["begin_counter"],
                         "closed service queue crosses callers or has concurrent submissions")
            queues[stream] = scope["thread"], row["end_counter"]
            expected[stream] += row["tasks"]
            observed.add((scope["id"], matches[0]))

        declared = {
            (scope["id"], index) for scope in services.values() for index in range(len(scope["service"]["executions"]))
        }
        self.require(observed == declared, "synchronous service executions lack raw queue evidence")
        self.require(
            not any(stream in expected and post.record["service"] < 0 for stream, post in self.queues.submissions),
            "private queue has unowned work outside its closed services",
        )

        for stream, count in expected.items():
            self.require(self.sdk.counts[stream] == count,
                         "private queue SDK coverage differs from its actual submitted SQE counts")

        self.require(
            not any(row["api"].startswith("aclrt") and row["stream_status"] == 0 and row["stream"] in expected
                    for row in self.runtime_records.values()),
            "closed service queue has external ACL operations",
        )

        return ({"synchronous_services": len(services), "executions": len(declared), "private_queues": len(expected),
                 "submitted_tasks": sum(expected.values())}, set(expected))

    @cached_property
    def staged_waits(self):
        first = {}
        for scope in self.native["scopes"]:
            for port in scope["ports"]:
                if port["direction"] != "input" or "producer" not in port:
                    continue
                producer = port["producer"]
                if self.native["scopes"][producer].get("local_exchanges"):
                    returned = scope["device_return"]
                    first[producer] = min(first.get(producer, returned), returned)
        return first

    def check_boundary(self, graph, event, resource, record, side, scope):
        measured = graph["evidence"]["events"][event]
        self.require(
            graph["events"][event]["resource"] == resource
            and measured["order"] == record[side + "_counter"]
            and measured["time_ns"] == record[side + "_raw_ns"],
            f"scope {scope} changes its native {side} boundary or measurement",
        )
