"""Check ranked execution against the physical compute streams and collective gates."""

from collections import deque

from prologue.acquisition.observation.batches import ObservedBatches, aligned_batches, aligned_collectives
from prologue.integrations.platforms.ascend.collectives import PhysicalCollectives
from prologue.verification.structure import StructureComparison


class RecordedParallelBatch:
    def __init__(self, profiles, batches, publisher):
        self.begin_ns = min(batch.call["begin_ns"] for batch in batches)
        self.end_ns = max(batch.call["result_processing"]["end_ns"] for batch in batches)
        self.calls = [{
            "operation": "parallel_batch", "implementation": "parallel_batch",
            "parameters": {"ranks": list(range(len(batches))), "publication_rank": publisher},
            "inputs": [], "outputs": [], "tensor_links": [], "path": [],
        }]
        self.work = []
        self.groups = []
        self.measurements = []
        self.tail = [None] * len(batches)
        pending = []
        for rank, batch in enumerate(batches):
            self.calls.extend({**call, "path": [rank, *call["path"]]} for call in batch.graph["calls"])
            cuts = {scope for scope, _ in batch.collectives()}
            pending.append(deque((node, row) for node, row in zip(batch.graph["work"], batch.device, strict=True)
                                 if row["owner"] not in cuts))

        for entries in aligned_collectives(batches):
            gates = [profile.gates.get(scope) for profile, (scope, _) in zip(profiles, entries, strict=True)]
            if None in gates:
                raise ValueError("the physical reference is missing a collective stream gate")
            for rank, gate in enumerate(gates):
                self.take_local(rank, pending[rank], profiles[rank].offset_ns, until=gate.signal)
                if pending[rank] and pending[rank][0][1]["id"] < gate.wait:
                    raise ValueError("unmodeled compute work crosses a collective gate")

            # The producer and consumer sides come from the actual compute stream events.
            predecessors = [tail for tail in self.tail if tail is not None]
            ready = max(gate.ready_raw_ns for gate in gates)
            kind = entries[0][1]["parameters"]["kind"]
            results = []
            for rank, ((_, call), gate) in enumerate(zip(entries, gates, strict=True)):
                path = [rank, *call["path"]]
                node = {"path": path, "kind": "collective", "name": kind, "ordinal": 0, "slot": 0,
                        "after": predecessors, "collective": len(self.groups)}
                results.append(self.append(rank, node, ready, gate.completed_raw_ns))
            self.groups.append({"kind": kind, "results": results})

        for rank, queue in enumerate(pending):
            self.take_local(rank, queue, profiles[rank].offset_ns)
        self.graph = {"calls": self.calls, "work": self.work, "collectives": self.groups}

    def take_local(self, rank, pending, offset_ns, *, until=None):
        while pending and (until is None or pending[0][1]["id"] < until):
            node, row = pending.popleft()
            local = {**node, "path": [rank, *node["path"]]}
            self.append(rank, local, row["begin_ns"] + offset_ns, row["end_ns"] + offset_ns)

    def append(self, rank, node, begin_ns, end_ns):
        if begin_ns < self.begin_ns or end_ns < begin_ns:
            raise ValueError("a measured parallel work interval violates its action or readiness")
        index = len(self.work)
        self.work.append({**node, "lane": rank})
        self.measurements.append({
            "submitted_after_ns": begin_ns - self.begin_ns, "dispatch_ns": 0, "duration_ns": end_ns - begin_ns,
        })
        self.tail[rank] = index
        return index


class ParallelStructureReference(StructureComparison):
    def __init__(self, recording):
        physical = PhysicalCollectives(recording)
        publisher = recording.publication_rank()
        manifest = {**recording.manifest, "delivery": recording.delivery}
        super().__init__(recording.domain, physical.ranks[publisher].publications, manifest)

        traces = [ObservedBatches(rank.evidence).entries for rank in physical.ranks]
        for batches in aligned_batches(traces):
            observed = RecordedParallelBatch(physical.ranks, batches, publisher)
            owner = batches[publisher]
            if any(batch.action() != owner.action() or batch.call["resources"] != owner.call["resources"]
                   or batch.call["result_processing"]["resources"] != owner.call["result_processing"]["resources"]
                   for batch in batches):
                raise ValueError("the TP ranks disagree on replicated request and resource state")
            self.record(owner.action(), observed.graph, owner.call, observed.measurements,
                        observed.begin_ns, observed.end_ns)

    @staticmethod
    def consumed_conditions(actual, measured):
        begin_ns = actual["began_ns"]
        for (start, end), sample in zip(actual["intervals"], measured["work"], strict=True):
            if start != begin_ns + sample["submitted_after_ns"] or end - start != sample["duration_ns"]:
                return False
        if any(row["at_ns"] != begin_ns + measured["publications"][row["request"]]["after_start_ns"]
               for row in actual["publications"]):
            return False
        return StructureComparison.consumed_conditions(actual, measured)
