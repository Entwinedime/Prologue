"""Qualify each physical collective endpoint against the posts of its actual peers."""

import json
from collections import defaultdict
from dataclasses import dataclass

from prologue.integrations.platforms.ascend.clock import DeviceCalibration
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

    def service(self, peers, rank):
        if self.elapsed is None:
            return None
        readiness = []
        for actor in self.required:
            if actor == rank:
                continue
            peer = peers[actor].ready
            if peer is None:
                return None

            # A peer known to precede our own start contributes no cross-device clock uncertainty.
            if peer[1] > self.ready[0]:
                readiness.append(peer)
        return RendezvousService.after_peers(self.elapsed, self.finish, readiness)


class CollectiveServices:
    def __init__(self, facts, capture, rank):
        channels = defaultdict(dict)
        measurements = {}
        for actor, document in facts.items():
            for channel, posts in self.posts(document):
                channels[channel][actor] = posts

            native = json.loads((capture / f"rank-{actor}.json").read_text())
            if native["environment"] != document["environment"]:
                raise ValueError("collective calibration and worker facts come from different environments")
            measurements[actor] = self.intervals(document, DeviceCalibration(native["device_clock"]))

        self.samples = {}
        for actors in channels.values():
            if rank not in actors:
                continue
            operations = [(operation, members) for _, operation, members in actors[rank]]
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
                own = peers[rank]
                if own.compatible(peers, members, operation, rank):
                    self.samples[own.position] = own.service(peers, rank), own.payload

    @staticmethod
    def intervals(document, clock):
        measurement = document["measurements"]
        points = {(row["resource"], row["order"]): row["time_ns"] for row in measurement["coordinates"]}
        peer_rows = measurement.get("collective_peer_rows", {})
        intervals = {}
        for scope, resource, begin, end, payload in measurement["collectives"]:
            if payload is None:
                continue
            start, finish = points[resource, begin], points[resource, end]
            intervals[scope] = CollectivePost(
                (resource, begin, end), payload, document["scopes"][scope]["communication"]["required"], peer_rows.get(str(scope)),
                clock.bounds(start) if start is not None else None, clock.bounds(finish) if finish is not None else None,
                clock.elapsed_bounds(start, finish) if start is not None and finish is not None else None,
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
