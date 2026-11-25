"""Resolve host activity, measured transfers and collective service costs."""

import json
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations

from prologue.common.cost import RendezvousService, ServiceSamples, cost_features, cost_identity
from prologue.common.observation import ThreadService
from prologue.integrations.platforms.ascend.analysis.trace import SynchronousCopy
from prologue.integrations.platforms.ascend.trace_store import (
    HOST_WAITS,
    DeviceCalibration,
    Interval,
    ThreadIntervals,
)


@dataclass(frozen=True, slots=True)
class NotifyProgress:
    prefix: tuple[int, int]
    suffix: tuple[int, int]

    @classmethod
    def within(cls, rows, start, finish, clock):
        endpoints = defaultdict(list)
        for side, local, remote, handle, stamp in rows:
            if not start <= stamp <= finish:
                raise ValueError("HCCL notification falls outside its measured collective")
            endpoints[side, local, remote, handle].append(
                cls(clock.elapsed_bounds(start, stamp), clock.elapsed_bounds(stamp, finish))
            )
        return endpoints

    @staticmethod
    def pairs(recorded, waited):
        for (side, local, remote, handle), steps in recorded.items():
            if side != "record" or local == remote:
                continue
            matches = waited.get(("wait", remote, local, handle), ())

            # Equal sequences propose a pairing. The calibrated endpoints qualify it for the channel.
            if len(steps) == len(matches):
                yield from zip(steps, matches, strict=True)


class ObservedServices:
    """Resolve measurements from the same observation by their actual resource intervals."""

    def __init__(self, graph, facts, collectives, *, physical_peers):
        self.graph, self.collectives = graph, collectives
        self.physical_peers = physical_peers
        measurement = facts["measurements"]
        if (graph["environment"] != facts["environment"]
                or graph["evidence"]["window_ns"] != measurement["window_ns"]):
            raise ValueError("cost measurements and coarse graph come from different native observations")

        self.host_positions = {}
        for lane, order, service_ns in measurement.get("host_service_positions", ()):
            if service_ns < 0 or (lane, order) in self.host_positions:
                raise ValueError("CPU service has a negative prefix or repeated position")
            self.host_positions[lane, order] = service_ns
        self.services = {}
        for lane, begin, end, duration_ns in measurement.get("device_services", ()):
            if duration_ns < 0 or (lane, begin, end) in self.services:
                raise ValueError("physical service has a negative duration or repeated position")
            self.services[lane, begin, end] = duration_ns
        self.transfers = {}
        for lane, begin, end, duration_ns, size, direction in measurement.get("synchronous_transfers", ()):
            if duration_ns < 0 or size <= 0 or (lane, begin, end) in self.transfers:
                raise ValueError("synchronous transfer has invalid or repeated service evidence")
            self.transfers[lane, begin, end] = duration_ns, size, direction
        self.payload = [resource["payload"] for resource in facts["resources"]]
        self.positions = [[begin for begin, _ in ranges] for ranges in self.payload]

    def host_service(self, piece):
        lane = piece["resource"]
        evidence = self.graph["evidence"]["events"]
        begin, end = (self.host_positions.get((lane, evidence[piece[key]]["order"])) for key in ("start", "finish"))
        if begin is None or end is None:
            return None
        if end < begin:
            raise ValueError("CPU service prefixes reverse work order")
        return end - begin

    def service(self, piece):
        lane = piece["resource"]
        evidence = self.graph["evidence"]["events"]
        begin, end = (evidence[piece[side]]["order"] for side in ("start", "finish"))
        if piece["kind"] == "collective":
            duration, payload = self.collectives.get((lane, begin, end), (None, None))
            return duration, None, payload
        if piece["kind"] == "communication":
            return None, None, None
        resource = self.graph["resources"][lane]
        if resource["kind"] != "stream":
            if piece["kind"] != "transfer":
                return None, None, None
            if resource["kind"] == "service" and not self.physical_peers:
                scope = self.graph["scopes"][piece["scope"]]

                # On one card this interval measures staging into another logical worker's pool.
                if any(access["pool"] != "local" for field in ("kv_accesses", "slot_accesses")
                       for access in scope[field]):
                    return None, None, None
            copy = self.transfers.get((lane, begin, end))
            if copy is None:
                return None, None, None
            duration, size, direction = copy
            if "copy" in piece:
                label = piece["copy"]["direction"]
                if direction == SynchronousCopy.directions[label]:
                    direction = label
            return duration, [size, direction], None

        positions, payload = self.positions[lane], self.payload[lane]
        first = bisect_left(positions, begin)
        last = bisect_left(positions, end, lo=first)
        if first == last or (first and payload[first - 1][1] > begin) or payload[last - 1][1] > end:
            raise ValueError("device work does not enclose complete native payload")
        durations = [self.services.get((lane, start, finish)) for start, finish in payload[first:last]]
        return (None if None in durations else sum(durations)), None, None


class HostCuts:
    def __init__(self, service):
        self.service = service
        self.points = {}
        self.misses = Counter()
        self.scopes = set()

    def point(self, thread, position, target_thread, target_position):
        key, mapped = (thread, position), (target_thread, target_position)
        if self.points.setdefault(key, mapped) != mapped:
            self.points[key] = None

    def anchors(self, source, target):
        self.point(source.thread, source.begin, target.thread, target.begin)
        self.point(source.thread, source.end, target.thread, target.end)

    def duration(self, thread, begin, end, submission):
        begin = self.points.get((thread, begin))
        end = self.points.get((thread, end))
        if begin is None or end is None or begin[0] != end[0]:
            return None
        return self.service.duration(begin[0], begin[1], end[1], submission)


class AscendHostAlignment:
    def __init__(self, trace, source_scopes, scopes, commands, runtime, activity, auxiliary, cold):
        self.trace = trace
        self.posts = {scope["id"]: scope["local_exchanges"] for scope in source_scopes if scope.get("local_exchanges")}
        self.observed = {scope["id"]: Interval(*scope["raw_interval"], scope["thread"], scope) for scope in scopes[1:]}

        # Unmatched children still own their records; they cannot transfer queue work to a matched parent.
        self.owners = ThreadIntervals(list(self.observed.values()))
        auxiliary = [Interval(begin, end, thread, {}) for thread, begin, end in auxiliary]
        excluded = ThreadIntervals(auxiliary)
        unavailable = list(cold)
        by_thread = ThreadService.by_thread(unavailable)
        filtered = []
        for command in commands:
            name, enqueue, dequeue = command
            entered = Interval(enqueue[1], enqueue[2], enqueue[0], {})
            if excluded.enclosing(entered) is not None:
                auxiliary.append(Interval(dequeue[1], dequeue[2], dequeue[0], {}))
            else:
                filtered.append(command)
                if ThreadService.within(by_thread.get(entered.thread, ()), entered.begin, entered.end):
                    unavailable.append(dequeue)
        excluded_time = [(row.thread, row.begin, row.end) for row in auxiliary]
        for row in runtime:
            thread = row["thread"]
            before, after = row["before_raw_ns"], row["after_raw_ns"]
            if row["api"] in HOST_WAITS or (row["api"] == "aclrtMemcpy" and row["bytes"]):
                excluded_time.append((thread, before, after))
            else:
                excluded_time.extend(((thread, before, row["begin_raw_ns"]), (thread, row["end_raw_ns"], after)))
        self.service = ThreadService(activity, (dequeue for _, _, dequeue in filtered), excluded_time, unavailable)
        runtime_intervals = []
        for row in runtime:
            call = Interval(row["begin_raw_ns"], row["end_raw_ns"], row["thread"], row)
            if row["result"] == 0 and excluded.enclosing(call) is None:
                runtime_intervals.append(call)
        self.runtime = ThreadIntervals(runtime_intervals)
        self.apis = HOST_WAITS | trace.stream_waits | {
            "aclrtMemcpy", "aclrtMemcpyAsync", "drvMemcpy", "halMemcpy", "halSqTaskSend",
        }
        self.previous, self.current = self.queues(filtered)

    @staticmethod
    def signature(call):
        row = call.record
        return row["api"], row.get("tasks", 0), row.get("bytes", 0), row.get("direction", -1)

    def calls(self, values):
        return [row for row in values if row.record["api"] in self.apis]

    def submission_cuts(self, cuts, declared, measured):
        timed = measured.record.get("local_submissions", ())
        operands = [(row["operation"], row["dtype"], row["bytes"]) for row in declared]
        if (operands != [(row["operation"], row["dtype"], row["bytes"]) for row in timed]
                or len(set(operands)) != len(operands)):
            cuts.misses["unpaired_local_submissions"] += len(declared)
            return

        # Both clocks locate the caller handing off the same operand. Peer staging itself stays excluded.
        for source, actual in zip(declared, timed, strict=True):
            origin = self.trace.device_points[source["begin"]].host
            stamp = actual["submitted_raw_ns"]
            if actual["thread"] != measured.thread or not measured.begin <= stamp <= measured.end:
                raise ValueError("local submission lies outside its timed native caller")
            cuts.point(origin.thread, origin.begin, measured.thread, stamp)

    def runtime_matches(self, cuts, old, new, strict):
        original = [self.signature(row) for row in old]
        measured = [self.signature(row) for row in new]
        if original != measured:
            cuts.misses["runtime_sequence"] += 1
            if strict:
                return None
            left, right = Counter(original), Counter(measured)
            shared = {key for key, count in left.items() if count == right[key]}

            # A transient runtime call can disappear after warmup. Pair only cuts with equal multiplicity and
            # preserved order.
            if [key for key in original if key in shared] != [key for key in measured if key in shared]:
                return ()
            old = [row for row, key in zip(old, original, strict=True) if key in shared]
            new = [row for row, key in zip(new, measured, strict=True) if key in shared]
        return zip(old, new, strict=True)

    def queues(self, commands):
        previous, current = defaultdict(list), defaultdict(list)
        for key, enqueue in sorted(self.trace.enqueue.items(), key=lambda row: row[1].begin):
            scope = self.trace.owner(enqueue)
            if scope is None or self.trace.excluded_ranges.enclosing(enqueue) is not None:
                continue
            dequeue = self.trace.dequeue[key]
            name = dequeue.record["name"].removeprefix("Dequeue@")
            previous[scope].append((name, enqueue, dequeue))
        for name, enqueue, dequeue in sorted(commands, key=lambda row: row[1][1]):
            entered = Interval(enqueue[1], enqueue[2], enqueue[0], {})
            owner = self.owners.enclosing(entered)
            if owner is not None:
                finished = Interval(dequeue[1], dequeue[2], dequeue[0], {})
                current[owner.record["id"]].append((name, entered, finished))
        return previous, current

    def cuts(self, pairs, queries, *, strict_runtime=False):
        result = HostCuts(self.service)
        for scope, target in pairs.items():
            old_scope = self.trace.scopes[scope]
            measured = self.observed[target]
            old, new = self.previous[scope], self.current[target]
            if [row[0] for row in old] != [row[0] for row in new]:
                result.misses["queue_sequence"] += 1
                continue
            old_runtime = [row for row in self.calls(self.trace.runtime_ranges.inside(old_scope))
                           if self.trace.owner(row) == scope and self.trace.excluded_ranges.enclosing(row) is None]
            new_runtime = [
                row for row in self.calls(self.runtime.inside(measured)) if self.owners.enclosing(row) is measured
            ]
            ranges = [(old_runtime, new_runtime)]
            enqueues = []
            for (_, entered, finished), (_, produced, consumed) in zip(old, new, strict=True):
                enqueues.extend(((entered, produced), (finished, consumed)))
                ranges.append((self.calls(self.trace.runtime_ranges.inside(finished)),
                               self.calls(self.runtime.inside(consumed))))
            runtime = [self.runtime_matches(result, before, after, strict_runtime) for before, after in ranges]
            if None in runtime:
                continue

            result.scopes.add(scope)
            result.anchors(old_scope, measured)
            if scope in self.posts:
                self.submission_cuts(result, self.posts[scope], measured)
            for source, target in enqueues:
                result.anchors(source, target)
            for matches in runtime:
                for source, target in matches:
                    result.anchors(source, target)

        for query in queries:
            scope = query["scope"]
            if scope in pairs and (not strict_runtime or scope in result.scopes):
                previous = self.trace.scopes[scope]
                result.anchors(Interval(query["begin_counter"], query["end_counter"], previous.thread, {}),
                               self.observed[pairs[scope]])
        return result


@dataclass(frozen=True, slots=True)
class CollectivePost:
    position: tuple[int, int, int]
    payload: list[int]
    required: list[int]
    peer_rows: dict | None
    ready: tuple[int, int] | None
    finish: tuple[int, int] | None
    elapsed: tuple[int, int] | None
    notifications: dict

    def operands(self, operation):
        dtype, algorithm, elements = self.payload

        # Transfer partners post opposite SDK operations against the same dtype and count.
        return (dtype, elements) if operation == "transfer" else (dtype, algorithm, elements)

    def compatible(self, peers, members, operation, rank):
        if self.peer_rows is None:
            expected = self.operands(operation)
            return all(peer.peer_rows is None and peer.operands(operation) == expected for peer in peers.values())

        local = members.index(rank)
        for index, actor in enumerate(members):
            peer = peers[actor]
            if (peer.peer_rows is None or peer.payload[:2] != self.payload[:2]
                    or self.peer_rows["send"][index] != peer.peer_rows["receive"][local]
                    or self.peer_rows["receive"][index] != peer.peer_rows["send"][local]):
                return False
        return True


class CollectiveClock:
    """Bound the measured endpoints of one rendezvous across its participants."""

    def __init__(self, peers):
        windows = {(rank, name): interval for rank, post in peers.items()
                   for name, interval in (("ready", post.ready), ("finish", post.finish)) if interval is not None}
        self.positions = {point: index for index, point in enumerate(windows)}

        # Each cell bounds time(later) - time(earlier); the two endpoints may be on different devices.
        self.upper = [[0 if i == j else end[1] - begin[0] for j, end in enumerate(windows.values())]
                      for i, begin in enumerate(windows.values())]
        for rank, post in peers.items():
            if post.elapsed is not None:
                self.constrain((rank, "ready"), (rank, "finish"), post.elapsed[1])
                self.constrain((rank, "finish"), (rank, "ready"), -post.elapsed[0])
        if not self.close():
            raise ValueError("measured collective endpoints contradict their calibrated elapsed times")

    def refine(self, peers):
        signals = [(sender, receiver, record, wait) for sender, receiver in permutations(peers, 2)
                   for record, wait in NotifyProgress.pairs(peers[sender].notifications, peers[receiver].notifications)]
        if not signals:
            return True

        measured = self.upper
        self.upper = [row.copy() for row in measured]
        for sender, receiver, record, wait in signals:
            self.notification(sender, receiver, record, wait)
        if self.close():
            return True
        self.upper = measured
        return False

    def close(self):

        # Compose calibrated and causal bounds through every observed post in this collective.
        for via, relay in enumerate(self.upper):
            for row in self.upper:
                to_relay = row[via]
                for target, current in enumerate(row):
                    bound = to_relay + relay[target]
                    if bound < current:
                        row[target] = bound
        return all(row[index] >= 0 for index, row in enumerate(self.upper))

    def constrain(self, before, after, bound):
        row, index = self.upper[self.positions[before]], self.positions[after]
        row[index] = min(row[index], bound)

    def notification(self, sender, receiver, record, wait):
        send_ready, send_finish = (sender, "ready"), (sender, "finish")
        receive_ready, receive_finish = (receiver, "ready"), (receiver, "finish")

        # The send can signal after its start; the wait can finish only after that signal.
        self.constrain(receive_ready, send_ready, wait.prefix[1] - record.prefix[0])
        self.constrain(receive_ready, send_finish, wait.prefix[1] + record.suffix[1])
        self.constrain(receive_finish, send_ready, -wait.suffix[0] - record.prefix[0])
        self.constrain(receive_finish, send_finish, record.suffix[1] - wait.suffix[0])

    def service(self, rank, required):
        finish = self.positions[rank, "finish"]
        starts = [self.positions[actor, "ready"] for actor in {rank, *required}]
        return RendezvousService.after_peers(
            [(-self.upper[finish][start], self.upper[start][finish]) for start in starts]
        )


class CollectiveServices:
    def __init__(self, facts, worker_sources, ranks):
        channels = defaultdict(dict)
        measurements = {}
        for actor, document in facts.items():
            for channel, posts in self.posts(document):
                channels[channel][actor] = posts

            native = json.loads((worker_sources[actor] / f"rank-{actor}.json").read_bytes())
            if native["environment"] != document["environment"]:
                raise ValueError("collective calibration and worker facts come from different environments")
            measurements[actor] = self.intervals(document, DeviceCalibration(native["device_clock"]))

        self.samples = {rank: {} for rank in ranks}
        self.notification_conflicts = {rank: {} for rank in ranks}
        for channel, actors in channels.items():
            selected = [rank for rank in ranks if rank in actors]
            if not selected:
                continue
            operations = [(operation, members) for _, operation, members in actors[selected[0]]]
            if any(actors.keys() != set(members) for _, members in operations) or any(
                [(operation, group) for _, operation, group in posts] != operations for posts in actors.values()
            ):
                continue
            self.channel_samples(channel, actors, operations, measurements, selected)

    def channel_samples(self, channel, actors, operations, measurements, selected):
        refined = {rank: {} for rank in selected}
        conflict = None

        # A reused notification register can invalidate the channel's round pairing as a whole.
        for ordinal, (operation, members) in enumerate(operations):
            scopes = {rank: actors[rank][ordinal][0] for rank in members}
            peers = {rank: measurements[rank].get(scope) for rank, scope in scopes.items()}
            if None in peers.values():
                continue
            aligned = {rank: post for rank, post in peers.items() if post.compatible(peers, members, operation, rank)}
            posts = {rank: aligned[rank] for rank in selected if rank in aligned}
            if not posts:
                continue
            clock = CollectiveClock(peers)
            measurable = {
                rank: post
                for rank, post in posts.items()
                if post.elapsed is not None and all(peers[actor].ready is not None for actor in post.required)
            }
            for rank, post in posts.items():
                service = clock.service(rank, post.required) if rank in measurable else None
                self.samples[rank][post.position] = service, post.payload
            if conflict is not None or len(aligned) != len(peers):
                continue
            if not clock.refine(peers):
                conflict = {"round": ordinal, "operation": operation, "source_scopes": scopes}
                continue
            for rank, post in measurable.items():
                service = clock.service(rank, post.required)
                if (service, post.payload) != self.samples[rank][post.position]:
                    refined[rank][post.position] = service, post.payload

        # Publish only measurements from the independent clock bounds when pairing contradicts those bounds.
        for rank in selected:
            if conflict is None:
                self.samples[rank].update(refined[rank])
            else:
                self.notification_conflicts[rank][channel] = conflict

    @staticmethod
    def intervals(document, clock):
        measurement = document["measurements"]
        points = {(row["resource"], row["order"]): row["time_ns"] for row in measurement["coordinates"]}
        notifications = measurement.get("collective_notifications", {})
        intervals = {}
        for scope, resource, begin, end, payload in measurement["collectives"]:
            if payload is None:
                continue
            start, finish = points[resource, begin], points[resource, end]
            group = document["scopes"][scope]["communication"]
            intervals[scope] = CollectivePost(
                (resource, begin, end),
                payload,
                group["required"],
                group.get("peer_rows"),
                clock.bounds(start) if start is not None else None,
                clock.bounds(finish) if finish is not None else None,
                clock.elapsed_bounds(start, finish) if start is not None and finish is not None else None,
                NotifyProgress.within(notifications.get(str(scope), ()), start, finish, clock)
                if start is not None and finish is not None
                else {},
            )
        return intervals

    @staticmethod
    def posts(document):
        channels = defaultdict(list)
        spans = document["spans"]
        for index, scope in enumerate(document["scopes"]):
            exchange = scope.get("communication")
            if exchange is None:
                continue
            entry = next((port["at"] for port in scope["ports"] if port["name"] == "call.enter"), None)
            lane, order = (None, None) if entry is None else (spans[entry["span"]]["resource"], entry["order"])
            channels[exchange["channel"]].append(
                (lane, order, index, exchange["operation"], tuple(exchange["members"]))
            )

        # A single observed issuing thread establishes the channel's order independently of batch annotations.
        for channel, posts in channels.items():
            lanes = {lane for lane, _, _, _, _ in posts}
            orders = {order for _, order, _, _, _ in posts}
            if len(lanes) != 1 or None in lanes or len(orders) != len(posts):
                continue
            lane, = lanes
            if document["resources"][lane]["kind"] != "thread":
                continue
            yield channel, [(index, operation, members) for _, _, index, operation, members in sorted(posts)]


class AscendTransportProfile:
    # HcclDataType is the native payload ABI. Names in tensor ports come from PyTorch.
    dtypes = ("int8", "int16", "int32", "float16", "float32", "int64", "uint64", "uint8",
              "uint16", "uint32", "float64", "bfloat16")
    uniform = {"all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor",
               "broadcast", "transfer"}

    def __init__(self, worker_sources, peers, trace, observation, image, deployment):
        self.qualification = {"observation": observation, "source_image": image}
        placement = {}
        for actor in peers:
            device, = (worker_sources[actor] / "native" / f"rank-{actor}").glob("*/PROF_*/device_*")
            number = int(device.name.removeprefix("device_"))
            info = json.loads((device / f"info.json.{number}").read_text())
            placement[str(actor)] = {"host": info["hostname"], "device": number, "driver": info["drvVersion"]}
        if len({(p["host"], p["device"]) for p in placement.values()}) != len(peers):
            raise ValueError("external transport calibration requires distinct physical peers")
        self.qualification["placement"] = placement
        self.qualification["logical_workers"] = {
            str(actor): doc["environment"]["parallel"] for actor, doc in peers.items()
        }

        # SDK link ranks are communicator-local. Keep the observed link types separate from the physical rank map.
        links = set()
        for event in json.loads(trace.read_text()):
            args = event.get("args", {})
            link = args.get("link type")
            if link and link not in {"ON_CHIP", "INVALID_TYPE"}:
                links.add((link, args["transport type"]))
        self.qualification["observed_links"] = sorted(links)
        if "transfer_protocol" in deployment:
            self.qualification["state_transfer_protocol"] = deployment["transfer_protocol"]

    @staticmethod
    def state_transfer(catalog, services, piece):
        scope = catalog.graph["scopes"][piece["scope"]]
        accesses = scope["kv_accesses"] + scope["slot_accesses"]
        pools = defaultdict(set)
        for access in accesses:
            if (access["start"], access["finish"]) == (piece["start"], piece["finish"]):
                pools[access["access"]].add(access["pool"])
        if pools["read"] != {"local"} or len(pools["write"]) != 1 or "local" in pools["write"]:
            raise ValueError("physical state transfer requires one remote destination")

        duration, copy, _ = services.service(piece)
        if copy is not None and copy[0] != scope["parameters"]["transfer_bytes"]:
            raise ValueError("measured state transfer changes payload bytes")
        return duration, copy

    def samples(self, catalog, services):
        source = ServiceSamples(catalog.domain)
        absent = Counter()
        receipts = defaultdict(lambda: defaultdict(list))
        coordinates, definitions = {}, {}
        for piece, definition, interval, features in catalog:
            native = catalog.definitions[definition]
            if ServiceSamples.state_transfer(native, interval):
                if "state_transfer_protocol" not in self.qualification:
                    raise ValueError("a physical state transfer profile requires its measured protocol")
                duration, copy = self.state_transfer(catalog, services, piece)
                if duration is None:
                    absent["unmeasured_state_transfer"] += 1
                    continue
                if definition not in definitions:
                    definitions[definition] = len(source.definitions)
                    source.definitions.append(native)
                source.add(definitions[definition], interval, features, duration, copy)
                continue
            if piece["kind"] != "collective":
                continue
            duration, _, payload = services.service(piece)
            if duration is None:
                reason = "unmeasured_peer_service" if payload is None else "unqualified_endpoint_clock"
                absent[reason] += 1
                continue
            identity = features.get("transport")
            if identity is None:
                absent["undeclared_operand_geometry"] += 1
                continue
            operation = identity["operation"]
            row_axes = identity.get("row_axes", ())
            if operation not in self.uniform and not row_axes:
                absent["requires_peer_payload_distribution"] += 1
                continue

            # HCCL reports the receive count for reduce-scatter and the posted buffer for send/receive.
            receives = operation in {"reduce_scatter", "reduce_scatter_tensor"} or (
                operation == "transfer" and identity["rank"] != identity["members"][0])
            ports = identity["ports"].get("output" if receives else "input", {}).values()
            dtype, _, elements = payload
            axes = features["axes"]
            operands = {(port["dtype"], axes[port["elements_axis"]]) for port in ports}
            if not 0 <= dtype < len(self.dtypes):
                absent["sdk_operand_mismatch"] += 1
                continue
            native_dtype = f"torch.{self.dtypes[dtype]}"

            # For variable exchanges the scalar SDK count cannot identify the payload split.
            compatible = (
                {name for name, _ in operands} == {native_dtype} if row_axes else operands == {(native_dtype, elements)}
            )
            if not compatible:
                absent["sdk_operand_mismatch"] += 1
                continue
            payload_axes = {port["elements_axis"]: axes[port["elements_axis"]]
                            for ports in identity["ports"].values() for port in ports.values()}
            payload_axes.update((axis, axes[axis]) for axis in row_axes)
            key = cost_identity((identity, payload_axes))
            coordinates[key] = identity, cost_features(payload_axes)
            receipts[key][tuple(payload)].append(duration)

        for key, variants in receipts.items():
            if len(variants) != 1:
                absent["ambiguous_native_implementation"] += sum(map(len, variants.values()))
                continue
            (payload, durations), = variants.items()
            identity, features = coordinates[key]
            for duration in durations:
                source.add_external(identity, "collective", features, duration, collective=payload)

        return source, absent
