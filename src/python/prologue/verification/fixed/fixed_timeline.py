"""Construct the measured blocking and service paths for the C++ replay engine."""

from dataclasses import dataclass

from prologue.verification.core.cpu_protocol import NativeCpuCall

from .fixed_context import CpuReadinessWake, ExternalWake, LocalCallWake, WorkProgress


@dataclass(frozen=True, slots=True)
class ServiceInterval:
    rank: int
    index: int | None
    start: int
    finish: int
    kind: str
    service_ns: int | None
    basis: str
    cuts: tuple[WorkProgress, ...]
    cpu_call: NativeCpuCall | None = None
    cpu_operation: str | None = None


@dataclass(frozen=True, slots=True)
class CpuTransportInterval:
    post: NativeCpuCall
    consumer: tuple[int, int] | None
    posts: tuple[tuple[int, int], ...]
    execution: dict
    window_ns: tuple[int, int]
    service_ns: int

    kind = "transfer"
    basis = "observed_cpu_transport_readiness"
    cuts = ()


@dataclass(frozen=True, slots=True)
class FixedDeviceRecord:
    post: int | WorkProgress
    duration_ns: int
    finished_ns: int


class FixedTimeline:
    def __init__(self, reference):
        self.reference = reference
        self.indices = reference.case.graph.positions
        self.releases = [reference.entries.get(event, 0) for event in self.indices]
        self.dependencies, self.work = [], []
        self.protocol = {}

        for post, (consumer, ready) in reference.cpu_initial_readiness.items():
            endpoint = self.protocol[post] = self.event(max(0, ready - reference.origin_ns))
            self.dependencies.append((endpoint, self.indices[consumer], 0))
        self.protocol.update((interval.post.posted, self.event()) for interval in reference.cpu_transfers)
        self.post_frontiers = {}
        released = set(reference.active_input_reads)
        released.update(wait.take_entry for wait in reference.host_waits.values() if wait.take_entry is not None)
        self.progress = {
            (rank, cut): self.event(max(0, cut.observed_ns - reference.origin_ns) if cut in released else 0)
            for (rank, _, _), cuts in reference.progress_cuts.items()
            for cut in cuts
        }
        self.input_progress = {cut: self.event() for cut in reference.active_input_reads}

    def event(self, release_ns=0):
        position = len(self.releases)
        self.releases.append(release_ns)
        return position

    def resume(self, rank, at, wait):
        resumed = self.event()
        waker = wait.source
        self.dependencies.append((at, resumed, wait.delay_ns if waker is None else 0))

        if waker is None:
            return resumed
        if isinstance(waker, ExternalWake):
            ready = waker.raw_ns - self.reference.origin_ns + wait.delay_ns
            self.releases[resumed] = max(self.releases[resumed], ready)
            return resumed

        if isinstance(waker, CpuReadinessWake):
            source = self.protocol[rank, waker.post]
        elif isinstance(waker, WorkProgress):
            source = self.progress[waker.rank, waker]
        elif isinstance(waker, LocalCallWake):
            source = self.event()
            entered = self.indices[waker.rank, waker.start]
            returned = self.indices[waker.rank, waker.finish]

            # The local wake follows call entry and precedes return, independently of peer readiness.
            self.dependencies.extend(((entered, source, waker.elapsed_ns), (source, returned, 0)))
        else:
            source = self.indices[waker]

        self.dependencies.append((source, resumed, wait.delay_ns))
        return resumed

    def transport(self, interval):
        frontier = self.post_frontiers.get(interval.posts)
        if frontier is None:
            frontier = self.post_frontiers[interval.posts] = self.event()
            self.dependencies.extend((self.indices[post], frontier, 0) for post in interval.posts)
        ready = self.protocol[interval.post.posted]
        self.work.append((frontier, ready, 0))
        if interval.consumer is not None:
            self.dependencies.append((ready, self.indices[interval.consumer], 0))

    def path(self, rank, start, finish, delay, cuts, *, modeled):
        edges = self.work if modeled else self.dependencies
        fixed_prefix = 0
        for cut in cuts:
            at = self.progress[rank, cut]
            reached = self.input_progress.get(cut, at)
            edges.append((start, reached, cut.fixed_prefix_ns - fixed_prefix))

            # Check local progress before the measured delivery can release the successful read.
            if reached != at:
                self.dependencies.append((reached, at, 0))
            start, fixed_prefix = at, cut.fixed_prefix_ns
            if wait := self.reference.host_waits.get((rank, cut)):
                start = self.resume(rank, at, wait)

        edges.append((start, finish, delay - fixed_prefix))

    def document(self):
        return {"releases_ns": self.releases, "dependencies": self.dependencies, "work": self.work}

    def build(self):
        reference = self.reference
        graph = reference.case.graph
        indices = self.indices
        fixed = {
            (indices[source], indices[target]): sum(conditions.values())
            for (source, target), conditions in reference.host_conditions.items()
        }
        order_cuts = reference.progress_cuts.copy()

        for interval in reference.intervals:
            if isinstance(interval, CpuTransportInterval):
                self.transport(interval)
                continue

            rank, begin, end = interval.rank, interval.start, interval.finish
            start, finish = indices[rank, begin], indices[rank, end]
            delay = fixed.pop((start, finish), 0)
            order_cuts.pop((rank, begin, end), None)

            if interval.kind == "collective":
                start = self.event()
                readiness = reference.peers[rank, end] | {(rank, begin)}
                self.dependencies.extend((indices[predecessor], start, 0) for predecessor in readiness)

            self.path(rank, start, finish, delay, interval.cuts, modeled=True)

        # Fences and device dispatches add measured handling after their causal frontier.
        gates = {}
        for target, tail in (reference.device_conditions | reference.host_fences).items():
            gate = self.event()
            self.dependencies.append((gate, indices[target], tail))
            gates[target] = gate

        for target, source in reference.signal_submissions.items():
            self.dependencies.append((indices[source], gates[target], 0))

        # Unpublished records use their measured submission and stream predecessor on the fixed self.
        for target, (previous, records) in reference.device_records.items():
            rank, _ = target
            predecessor = indices[rank, previous]
            for record in records:
                post = (
                    self.progress[rank, record.post]
                    if isinstance(record.post, WorkProgress)
                    else indices[rank, record.post]
                )
                complete = self.event()
                self.dependencies.extend(
                    ((post, complete, record.duration_ns), (predecessor, complete, record.duration_ns))
                )
                predecessor = complete

            self.dependencies.append((predecessor, gates[target], 0))

        # Fixed delays at regular dependencies also include measured handoffs between distinct host threads.
        for source, targets in graph.next.items():
            for target in targets:
                pair = indices[source], indices[target]
                delay = fixed.pop(pair, 0)
                cuts = order_cuts.pop((source[0], source[1], target[1]), ()) if source[0] == target[0] else ()
                self.path(source[0], pair[0], gates.get(target, pair[1]), delay, cuts, modeled=False)

        if fixed or order_cuts:
            raise ValueError("fixed timing condition does not belong to a verified dependency")
        self.input_positions = {
            cut: (reached, self.progress[cut.rank, cut]) for cut, reached in self.input_progress.items()
        }

        return self.document()

    def compare_active_input_reads(self, replayed, *, measured=False):
        differences = []
        waiting_ns, delayed = 0, 0
        for cut, (progress, delivered) in self.input_positions.items():
            observed = cut.observed_ns - self.reference.origin_ns
            reached, actual = replayed[progress], replayed[delivered]
            waiting_ns += actual - reached
            delayed += actual > reached

            if actual != max(observed, reached) or (measured and reached != observed):
                differences.append(
                    {
                        "rank": cut.rank,
                        "requests": self.reference.active_input_reads[cut],
                        "observed_ns": observed,
                        "local_progress_ns": reached,
                        "replayed_ns": actual,
                    }
                )

        return {
            "checked": len(self.input_positions),
            "delayed_deliveries": delayed,
            "total_delivery_wait_ns": waiting_ns,
            "different": len(differences),
            "differences": differences,
        }

    def transport_baseline(self, replayed):
        report = {
            "checked": len(self.reference.cpu_transfers),
            "initial_conditions": len(self.reference.cpu_initial_readiness),
            "different": 0,
        }

        for interval in self.reference.cpu_transfers:
            stamp = self.reference.origin_ns + replayed[self.protocol[interval.post.posted]]
            begin, end = interval.window_ns
            if not begin <= stamp <= end:
                if not report["different"]:
                    report["first_difference"] = {**self.reference.services.detail(interval), "replayed_raw_ns": stamp}
                report["different"] += 1

        return report

    def replay_services(self, durations):
        steps = []
        for interval, duration in zip(self.reference.intervals, durations, strict=True):
            previous = 0
            for cut in interval.cuts:

                # Fixed scenarios preserve the measured progress at each intra-work boundary.
                prefix = duration * cut.service_prefix_ns // interval.service_ns
                steps.append(prefix - previous)
                previous = prefix
            steps.append(duration - previous)
        return steps
