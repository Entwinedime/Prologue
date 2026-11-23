"""Read physical collective readiness and completion from Ascend traces."""

import ast
import json
from collections import defaultdict
from dataclasses import dataclass

from prologue.integrations.platforms.ascend.evidence import AscendEvidence


@dataclass(frozen=True, slots=True)
class CollectiveGate:
    ready_raw_ns: int
    completed_raw_ns: int
    signal: int
    wait: int


class PhysicalRank:
    def __init__(self, recording, rank):
        observation = recording.read(rank)
        self.evidence = observation.document()
        self.publications = observation.publications
        self.directory = recording.profile_directory(rank)
        self.offset_ns = self.clock_offset()

    def clock_offset(self):
        start, = self.directory.rglob("host/start_info")
        stamp = ast.literal_eval(start.read_text(encoding="utf-8"))

        # CANN uses this pair when exporting profiler timestamps in realtime.
        offset = int(stamp["clockMonotonicRaw"]) - int(stamp["collectionTimeBegin"]) * 1000
        calls = self.evidence["calls"]
        starts = self.evidence["profile_scope_begins_ns"]
        ends = self.evidence["profile_scope_ends_ns"]
        for call, begin, end in zip(calls, starts, ends, strict=True):
            if begin + offset > call["begin_ns"] or end + offset < call["end_ns"]:
                raise ValueError("the Ascend clock anchor does not enclose the observed calls")
        return offset

    def collective_gates(self):
        calls = self.evidence["calls"]
        work = self.evidence["work"]
        lane, = {row["lane"] for row in work if calls[row["owner"]]["operation"] != "collective"}
        by_scope = defaultdict(list)
        for row in work:
            if row["lane"] == lane and calls[row["owner"]]["operation"] == "collective":
                by_scope[row["owner"]].append(row)

        result = {}
        for scope, rows in by_scope.items():
            if [row["kind"] for row in rows] == ["EVENT_RECORD", "EVENT_WAIT"]:
                record, wait = rows
                result[scope] = CollectiveGate(record["end_ns"] + self.offset_ns, wait["end_ns"] + self.offset_ns,
                                               record["id"], wait["id"])
        return result

    def transport_links(self):
        matrix = self.directory / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
        groups = json.loads(matrix.read_text(encoding="utf-8"))["step"]["collective"]
        for links in groups.values():
            for key, value in links.items():
                source, target = map(int, key.split("-"))
                if source != target:
                    yield source, target, value["Transport Type"]


class PhysicalCollectives:
    def __init__(self, recording):
        if recording.windows is None or recording.manifest["placement"] != "full" or len(recording.windows) != 1:
            raise ValueError("physical collectives require a simultaneous full-rank recording")
        self.ranks = [PhysicalRank(recording, rank) for rank in recording.bindings()]
        for rank in self.ranks:
            rank.gates = rank.collective_gates()
        self.hardware = AscendEvidence.hardware(self.ranks[0].directory)
        if any(AscendEvidence.hardware(rank.directory) != self.hardware for rank in self.ranks[1:]):
            raise ValueError("the collective recording has different accelerators or native software across ranks")

        links = set()
        for rank in self.ranks:
            links.update(rank.transport_links())
        if not links:
            raise ValueError("the collective recording has no measured physical links")
        self.transport_links = [
            {"source": source, "target": target, "transport": transport}
            for source, target, transport in sorted(links)
        ]
