"""Qualify each physical collective endpoint against the posts of its actual peers."""

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

from prologue.integrations.platforms.ascend.clock import DeviceCalibration
from prologue.integrations.platforms.ascend.notifications import NotifyProgress
from prologue.performance.samples import RendezvousService


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

        for sender, receiver in permutations(peers, 2):
            for record, wait in NotifyProgress.pairs(peers[sender].notifications, peers[receiver].notifications):
                self.notification(sender, receiver, record, wait)

        # Compose calibrated and causal bounds through every observed post in this collective.
        for via, relay in enumerate(self.upper):
            for row in self.upper:
                to_relay = row[via]
                for target, current in enumerate(row):
                    bound = to_relay + relay[target]
                    if bound < current:
                        row[target] = bound
        if any(row[index] < 0 for index, row in enumerate(self.upper)):
            raise ValueError("HCCL notification order contradicts the calibrated worker clocks")

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
        return RendezvousService.after_peers([(-self.upper[finish][start], self.upper[start][finish]) for start in starts])


class CollectiveServices:
    def __init__(self, facts, capture, ranks):
        channels = defaultdict(dict)
        measurements = {}
        for actor, document in facts.items():
            for channel, posts in self.posts(document):
                channels[channel][actor] = posts

            native = json.loads((capture / f"rank-{actor}.json").read_text())
            if native["environment"] != document["environment"]:
                raise ValueError("collective calibration and worker facts come from different environments")
            measurements[actor] = self.intervals(document, DeviceCalibration(native["device_clock"]))

        self.samples = {rank: {} for rank in ranks}
        for actors in channels.values():
            selected = [rank for rank in ranks if rank in actors]
            if not selected:
                continue
            operations = [(operation, members) for _, operation, members in actors[selected[0]]]
            members = operations[0][1]
            if actors.keys() != set(members) or any(
                [(operation, group) for _, operation, group in posts] != operations for posts in actors.values()
            ):
                continue

            # Align complete native post sequences before qualifying the evidence for each round.
            for ordinal, (operation, _) in enumerate(operations):
                peers = {actor: measurements[actor].get(actors[actor][ordinal][0]) for actor in members}
                if None in peers.values():
                    continue
                clock = None
                for rank in selected:
                    own = peers[rank]
                    if not own.compatible(peers, members, operation, rank):
                        continue
                    service = None
                    if own.elapsed is not None and all(peers[actor].ready is not None for actor in own.required):
                        if clock is None:
                            clock = CollectiveClock(peers)
                        service = clock.service(rank, own.required)
                    self.samples[rank][own.position] = service, own.payload

    @staticmethod
    def intervals(document, clock):
        measurement = document["measurements"]
        points = {(row["resource"], row["order"]): row["time_ns"] for row in measurement["coordinates"]}
        peer_rows = measurement.get("collective_peer_rows", {})
        notifications = measurement.get("collective_notifications", {})
        intervals = {}
        for scope, resource, begin, end, payload in measurement["collectives"]:
            if payload is None:
                continue
            start, finish = points[resource, begin], points[resource, end]
            intervals[scope] = CollectivePost(
                (resource, begin, end), payload, document["scopes"][scope]["communication"]["required"], peer_rows.get(str(scope)),
                clock.bounds(start) if start is not None else None, clock.bounds(finish) if finish is not None else None,
                clock.elapsed_bounds(start, finish) if start is not None and finish is not None else None,
                NotifyProgress.within(notifications.get(str(scope), ()), start, finish, clock)
                if start is not None and finish is not None else {},
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
            channels[exchange["channel"]].append((lane, order, index, exchange["operation"], tuple(exchange["members"])))

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
