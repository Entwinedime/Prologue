"""Bind same-run service evidence and request boundaries to observed execution."""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import chain

from prologue.integrations.platforms.ascend.services import ObservedServices
from prologue.integrations.runtimes.pytorch.protocol_work import CpuProtocolWork

from .cpu_protocol import NativeCpuCall
from .fixed_context import CpuReadinessWake, ExternalWake, KernelWake, LocalCallWake, WorkProgress
from .ingress import IngressEvidence
from .publications import FixedPublications
from .source_clock import ClockRealization, CounterPoint


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
        self.progress = {(rank, cut): self.event(max(0, cut.observed_ns - reference.origin_ns) if cut in released else 0)
                         for (rank, _, _), cuts in reference.progress_cuts.items() for cut in cuts}
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


class FixedReference:
    def __init__(self, case):
        self.case = case
        self.peers = {}
        for source, target in case.connections.native_communication_edges:
            self.peers.setdefault(target, set()).add(source)
        self.gaps = {}
        self.input_releases = {}
        self.active_input_reads = {}
        self.requests = self.align_requests()
        self.cpu_calls = [call for calls in case.cpu.native_calls.values() for call in calls if call.execution["in_flow"]]
        self.cpu_points = {point: call for call in self.cpu_calls
                           for point in (call.entered, call.posted, call.returned) if point is not None}
        self.external_cpu_points = case.cpu.native_times.keys() - self.cpu_points.keys()
        self.specialization_loads = {rank: raw.get("specialization_loads") for rank, (_, raw, _) in case.sources.items()}

        dispatches = {(rank, start): dispatch for rank, record in case.measurements.items()
                      for start, dispatch in record["device_releases"].items()}
        self.clock = ClockRealization(case.clocks, case.graph, case.cpu.native_times)
        for target, dispatch in dispatches.items():
            submitted = target[0], dispatch.submitted

            # A native signal's driver submission precedes the physical command entry.
            if dispatch.submitted_at_completion and target not in case.graph.next[submitted]:
                self.clock.constrain_order(submitted, target)
        self.clock.solve()
        self.host_waits = dict(self.input_releases)
        self.host_waits.update(((rank, wait.stopped), wait) for rank, record in case.measurements.items()
                               for wait in record["fixed_host_waits"])

        # Match measured protocol readiness before assigning the remaining wakes to their producer lanes.
        self.cpu_initial_readiness = {}
        self.cpu_transfers = self.bind_cpu_transfers()
        self.bind_thread_wakes()
        self.observer_joins = {(rank, event) for rank, record in case.measurements.items()
                               for event in record["observer_joins"]}
        self.elided_waits = {(rank, event) for rank, events in case.logical.items() for event in events}
        fences = [(rank, source, target) for rank, record in case.measurements.items()
                  for source, target in sorted(record["host_fences"])]
        signals = [(rank, start, row.completion) for (rank, start), row in dispatches.items()
                        if row.completion is not None and (rank, row.completion) not in self.observer_joins]
        completions = {(rank, target) for rank, _, target in chain(fences, signals)}
        joins = self.observer_joins | self.elided_waits
        predecessors = self.predecessors(joins | dispatches.keys() | completions)
        self.event_windows = self.completion_windows(predecessors)
        self.publications = FixedPublications(case, self.requests, self.origin_ns)
        if missing := self.publications.coverage_gaps():
            self.gaps["incomplete_publication_boundaries"] = missing
        self.host_conditions = {((rank, source), (rank, target)): conditions for rank, record in case.measurements.items()
                                for (source, target), conditions in record["fixed_host_delays"].items()}
        self.host_fences, self.fence_evidence = self.bind_completions(
            fences, predecessors, "host_fence_timing_unqualified"
        )
        self.device_records = {}
        self.device_conditions, self.signal_submissions, self.device_dispatch = self.bind_device_dispatches(
            dispatches, predecessors, joins
        )
        signal_conditions, self.device_dispatch["signal_completions"] = self.bind_completions(
            signals, predecessors, "stream_signal_timing_unqualified"
        )
        self.device_conditions.update(signal_conditions)
        self.progress_cuts = self.collect_progress_cuts()
        self.intervals = [*self.services(), *self.cpu_transfers]
        self.entries = self.entry_releases()
        unbound = {rank: record["unbound_queue_notifications"] for rank, record in case.measurements.items()
                   if record["unbound_queue_notifications"]}
        if unbound:
            self.gaps["unbound_queue_notifications"] = unbound

    def bind_thread_wakes(self):
        owners = {kernel: (rank, thread) for rank, record in self.case.measurements.items()
                  for thread, kernel in record["kernel_threads"].items()}
        self.cross_worker_wakes = 0
        self.post_graph_host_wakes = 0
        for (rank, stopped), wait in tuple(self.host_waits.items()):
            witness = wait.source
            if not isinstance(witness, KernelWake):
                continue
            location = (rank, witness.local_thread) if witness.local_thread is not None else owners.get(witness.kernel_thread)
            if location is None:
                self.host_waits[rank, stopped] = replace(wait, source=ExternalWake(witness.raw_ns))
                continue

            owner, thread = location
            context = self.case.measurements[owner]["fixed_host_context"]
            call = context.call_at(thread, witness.raw_ns)
            if call is None:
                anchor, tail = context.serial_wake_anchor(thread, witness.raw_ns)
            else:
                anchor, tail = self.call_wake(context, call, witness.raw_ns), 0
            if anchor is None or (isinstance(anchor, (WorkProgress, LocalCallWake))
                                  and (owner, anchor.finish) not in self.case.graph.next[owner, anchor.start]):
                self.gaps.setdefault("unbound_worker_wakes", []).append([rank, stopped.start, stopped.finish, owner])
                del self.host_waits[rank, stopped]
                continue
            self.host_waits[rank, stopped] = replace(wait, source=anchor, delay_ns=wait.delay_ns + tail)
            self.cross_worker_wakes += owner != rank and not isinstance(anchor, ExternalWake)
            self.post_graph_host_wakes += isinstance(anchor, tuple) and tail > 0

    def call_wake(self, context, call, stamp):
        rank, piece = context.rank, call.record
        start, finish = piece["start"], piece["finish"]
        begin, end = call.begin, call.end
        graph = self.case.sources[rank][0].graph
        if graph["resources"][piece["resource"]]["kind"] == "thread":
            window = self.collective_window(rank, graph, piece, ())
            if window is None:
                return None
            begin, end = window
            if stamp < begin:

                # A pre-rendezvous wake follows the measured running prefix of this caller.
                if stamp > call.begin:
                    running = context.active.running_intervals(call.thread, call.begin, stamp)
                    if running is None or list(running) != [(call.begin, stamp)]:
                        return None
                return LocalCallWake(rank, start, finish, stamp - call.begin)

        return WorkProgress(rank, start, finish, stamp, stamp - begin, 0, end - begin)

    def bind_cpu_transfers(self):
        case = self.case
        posts = {(call.key, rank): call for rank, calls in case.cpu.native_calls.items()
                 for call in calls if call.phase == "post"}
        consumer_waits = defaultdict(list)
        for (rank, stopped), wait in self.host_waits.items():
            if (rank, stopped.finish) in self.cpu_points:
                consumer_waits[rank, stopped.finish].append(wait)
        transfers, unmeasured, unbound = [], [], []
        for key, peers in case.cpu.messages["native"].items():
            consumers = {rank: peer["consumed"] for rank, peer in peers.items() if peer["consumed"] in self.cpu_points}
            calls = {rank: posts[key, rank] for rank in peers}
            active = [call.execution for call in calls.values() if call.execution["in_flow"]]
            if not consumers and not active:
                continue
            timing = case.cpu.native_readiness(peers)
            checked = calls if active else consumers
            if timing is None:
                unmeasured.extend((key, rank) for rank in checked)
                continue

            posted, windows = timing
            if not active and posted >= self.origin_ns:
                unmeasured.extend((key, rank) for rank in checked)
                continue
            frontiers = tuple(call.posted for call in calls.values())
            for rank in checked:
                call = calls[rank]
                native = peers[rank]["transport"]
                if windows[rank] is None:
                    unmeasured.append((key, rank))
                    continue
                first, last = windows[rank]
                consumer = peers[rank]["consumed"]

                # All wakes issued by the completion callback follow its first observed readiness.
                kernel = case.measurements[rank]["kernel_threads"].get(native["completion_thread"])
                sleeps = consumer_waits[consumer]
                matches = [wait for wait in sleeps if isinstance(wait.source, KernelWake)
                           and wait.source.kernel_thread == kernel and kernel is not None
                           and first <= wait.source.raw_ns <= last]
                ready = first
                if matches:
                    ready = min(wait.source.raw_ns for wait in matches)
                    for wait in matches:
                        delay = wait.source.raw_ns - ready + wait.delay_ns
                        self.host_waits[rank, wait.stopped] = replace(wait, source=CpuReadinessWake(call.posted[1]), delay_ns=delay)
                elif sleeps and last >= case.cpu.native_times[self.cpu_points[consumer].entered]:
                    unbound.append((key, rank))

                # An already posted message contributes initial readiness without charging pre-window transfer work.
                if posted < self.origin_ns:
                    if consumer in self.cpu_points:
                        self.cpu_initial_readiness[call.posted] = consumer, ready
                else:
                    execution = call.execution if call.execution["in_flow"] else active[0]
                    transfers.append(CpuTransportInterval(call, consumer, frontiers, execution, (first, last), ready - posted))

        for name, missing in (("unmeasured_cpu_transport", unmeasured), ("unbound_cpu_transport_wakes", unbound)):
            if missing:
                key, rank = missing[0]
                self.gaps[name] = {"count": len(missing), "first": {"key": key, "rank": rank}}
        return transfers

    def collect_progress_cuts(self):
        cuts = defaultdict(set)
        for cut in self.active_input_reads:
            cuts[cut.rank, cut.start, cut.finish].add(cut)
        for (_, stopped), wait in self.host_waits.items():
            for cut in (stopped, wait.source, wait.take_entry):
                if isinstance(cut, WorkProgress):
                    cuts[cut.rank, cut.start, cut.finish].add(cut)
        for (rank, _), (_, records) in self.device_records.items():
            for record in records:
                post = record.post
                if isinstance(post, WorkProgress):
                    cuts[rank, post.start, post.finish].add(post)
        return {step: tuple(sorted(points, key=lambda point: point.observed_ns)) for step, points in cuts.items()}

    def predecessors(self, targets):
        parents = defaultdict(list)
        for source, following in self.case.graph.next.items():
            for target in following.keys() & targets:
                parents[target].append(source)
        return parents

    def bind_completions(self, waits, predecessors, gap):
        positions = self.case.graph.positions
        qualified, unqualified = {}, []
        counts = Counter()
        for rank, begin, end in waits:
            entry, returned = (rank, begin), (rank, end)
            parents = self.readiness_parents(returned, predecessors, self.elided_waits)
            if entry not in parents:
                unqualified.append((rank, begin, end))
                continue

            entered = self.clock.timestamp(*entry)
            finish = self.clock.timestamp(*returned)
            readiness = {parent: self.event_windows[positions[parent]] for parent in parents}
            if any(window is None or parent in self.observer_joins and window[1] > entered
                   for parent, window in readiness.items()):
                unqualified.append((rank, begin, end))
                continue

            # A verified completion follows the actual producer and the wait entry on either lane.
            ready = max(self.clock.timestamp(*parent) + sum(self.host_conditions.get((parent, returned), {}).values())
                        for parent in parents if parent not in self.observer_joins)
            if ready > finish:
                unqualified.append((rank, begin, end))
                continue

            qualified[returned] = finish - ready
            counts["frontier_in_call" if ready > entered else "ready_on_entry"] += 1
            counts["return_delay_ns"] += finish - ready
        if unqualified:
            self.gaps[gap] = {"count": len(unqualified), "first": unqualified[0]}
        return qualified, {"qualified": len(qualified), **counts}

    @staticmethod
    def readiness_parents(target, predecessors, joins):
        pending = list(predecessors[target])
        seen, physical = set(), set()

        # Follow the actual prerequisites of untimed joins while retaining their edges in the replay.
        for parent in pending:
            if parent in seen:
                continue
            seen.add(parent)
            if parent in joins:
                pending.extend(predecessors[parent])
            else:
                physical.add(parent)
        return physical

    def bind_omitted_records(self, rank, dispatch, parents):
        previous = rank, dispatch.previous
        if not dispatch.omitted or previous not in parents:
            return ()
        clock = self.case.clocks[rank]
        posts = self.case.measurements[rank]["fixed_driver_posts"]
        previous_finish = self.clock.timestamp(*previous)
        records = []
        for omitted in dispatch.omitted:
            host = posts.get(omitted)
            point = clock.point(omitted.completed_device_ns, True)
            earliest, _ = clock.point_window(point)
            finish = self.clock.raw_time(rank, point)
            ready = max(omitted.posted_ns, previous_finish)
            if host is None or earliest < omitted.posted_ns or ready > finish:
                return ()
            records.append(FixedDeviceRecord(host, finish - ready, finish))
            previous_finish = finish
        return tuple(records)

    def frontier_ready(self, rank, frontier, parents, completed=None):
        positions = self.case.graph.positions
        windows = [self.event_windows[positions[event]] for event in frontier]
        if None in windows:
            return False
        earliest = max(window[0] for window in windows)
        clock = self.case.clocks[rank]
        points = [clock.points[event] for _, event in frontier]
        if completed is not None:
            point = clock.point(completed, True)
            points.append(point)
            earliest = max(earliest, clock.point_window(point)[0])
        device = max((point.nominal_ns for point in points if isinstance(point, CounterPoint)), default=None)

        for parent in parents:
            if parent in frontier:
                continue
            window = self.event_windows[positions[parent]]
            if window is not None and window[1] <= earliest:
                continue

            # Ordering on a shared device counter survives overlapping RAW calibration windows.
            actor, event = parent
            point = clock.points[event] if actor == rank and event >= 0 else None
            if device is None or not isinstance(point, CounterPoint) or point.nominal_ns > device:
                return False
        return True

    def bind_device_dispatches(self, dispatches, predecessors, joins):
        positions = self.case.graph.positions
        delays, signal_submissions, counts = {}, {}, Counter()
        for (rank, start), dispatch in dispatches.items():
            target = rank, start
            if target in self.observer_joins:
                continue
            source = rank, dispatch.submitted
            parents = self.readiness_parents(target, predecessors, joins)
            previous = rank, dispatch.previous
            preceding = dispatch.preceding_window_ns
            posted = self.event_windows[positions[source]]
            observed_start = self.event_windows[positions[target]]
            if posted is None or observed_start is None:
                counts["readiness_unqualified"] += 1
                continue

            driver_entry = posted[0]
            records = ()
            if not dispatch.omitted and previous in parents:
                frontier = {source, previous}
                basis = "observed_preceding_task"
            elif dispatch.first_command:
                frontier = {source}
                basis = "first_observed_command"
            elif preceding is not None and preceding[1] <= driver_entry:
                frontier = {source}
                basis = "preceding_task_before_calibration" if preceding[0] is None else "preceding_task_already_complete"
            elif records := self.bind_omitted_records(rank, dispatch, parents):
                frontier = {source, previous}
                basis = "observed_unpublished_command"
            else:
                counts["stream_state_unqualified"] += 1
                continue

            completed = dispatch.omitted[-1].completed_device_ns if records else None
            if not self.frontier_ready(rank, frontier, parents, completed):
                counts["readiness_unqualified"] += 1
                continue

            # A completion-linked post still releases the outgoing command at its physical entry.
            missing_submission = source not in parents
            if missing_submission and not dispatch.submitted_at_completion:
                counts["readiness_unqualified"] += 1
                continue

            # Both the driver post and the stream predecessor remain prerequisites when the frontier changes.
            ready = max(self.clock.timestamp(*event) for event in frontier)
            if records:
                ready = max(ready, records[-1].finished_ns)
            duration = self.clock.timestamp(*target) - ready
            if duration < 0:
                counts["readiness_unqualified"] += 1
                continue

            if missing_submission:
                signal_submissions[target] = source
                counts["verified_signal_submissions"] += 1

            delays[target] = duration
            if records:
                self.device_records[target] = (dispatch.previous, records)
            counts[basis] += 1
        return delays, signal_submissions, {"candidates": len(dispatches), "attached": len(delays), **counts}

    def align_requests(self):
        case = self.case
        self.origin_ns, = {entry.origin_ns for entry in case.submissions}
        intake = []
        for instance, submitted in zip(case.manifest["instances"], case.submissions, strict=True):
            if instance["mode"] != "decode":
                intake.append([{**row, "arrival_ns": submitted.by_request[name]["submitted_ns"]}
                               for name, row in submitted.requests.items()])
            self.bind_ingress(instance, submitted)
        workload, = intake
        return workload

    def bind_ingress(self, instance, submitted):
        for rank in instance["native_participants"]:
            contents, raw, _ = self.case.sources[rank]
            if not raw["request_ingress"]:
                continue
            start, _ = contents.graph["evidence"]["window_ns"]
            host = self.case.measurements[rank]
            wakes = host["fixed_idle_inputs"]
            reads = {row["received_raw_ns"]: row for row in raw.get("request_input_attempts", ()) if row["requests"]}
            for read in submitted.receives(raw["request_ingress"]):
                received = self.origin_ns + read["time_ns"]
                if received <= start:
                    continue

                if cut := wakes.get(received):
                    self.input_releases[rank, cut.stopped] = cut
                    continue

                submission = self.origin_ns + max(submitted.by_request[name]["submitted_ns"] for name in read["requests"])
                attempt = reads.get(received)
                if (attempt is not None and attempt["context"] is not None
                        and not IngressEvidence.idle(attempt["context"]) and submission <= attempt["begin_raw_ns"]):
                    cut = host["fixed_host_context"].input_read_progress(attempt["begin_raw_ns"], received)
                    if cut is not None:
                        self.active_input_reads[cut] = read["requests"]
                        continue

                self.gaps.setdefault("unqualified_input_readiness", []).append({
                    "rank": rank, "requests": read["requests"], "submission_raw_ns": submission, "received_raw_ns": received,
                })

    def services(self):
        case = self.case
        for rank, (contents, raw, _) in case.sources.items():
            graph = contents.graph
            document = case.measurements[rank]
            measurements = document["measurements"]
            captured_collectives = {(lane, begin, end) for _, lane, begin, end, payload in measurements.get("collectives", ())
                                    if payload is not None}
            services = ObservedServices(graph, document, {}, physical_peers=case.joint_execution)
            isolated = case.measured_service_pairs(rank, graph, raw, resources=document["resources"])
            for index, piece in enumerate(graph["work"]):
                start, finish = piece["start"], piece["finish"]
                host = piece["kind"] == "host"
                if host:
                    duration = services.host_service(piece)
                    basis = "observed_host_occupancy" if duration is not None else "missing"
                elif piece["kind"] == "collective":
                    window = self.collective_window(rank, graph, piece, captured_collectives)
                    duration = window[1] - window[0] if window is not None else None
                    resource = graph["resources"][piece["resource"]]
                    basis = "observed_host_collective" if resource["kind"] == "thread" else "observed_device_collective"
                    if duration is None:
                        basis = "missing"
                else:
                    duration, _, _ = services.service(piece)
                    payload = document["device_payload_times"].get((start, finish))
                    if duration is not None and payload:
                        duration = sum(self.clock.device_duration(rank, first, last) for first, last in payload)
                    basis = "observed_payload" if duration is not None else "missing"
                if duration is None and not host and (start, finish) in isolated:
                    duration = self.clock.timestamp(rank, finish) - self.clock.timestamp(rank, start)
                    basis = "isolated_observed_interval"

                cuts = self.progress_cuts.get((rank, start, finish), ())
                if any(cut.service_ns != duration for cut in cuts):
                    self.gaps.setdefault("unqualified_work_service_partitions", []).append((rank, start, finish))
                yield ServiceInterval(rank, index, start, finish, piece["kind"], duration, basis, cuts)

        for call in self.cpu_calls:
            rank = call.entered[0]
            for start, finish, operation in call.services():
                duration = case.measurements[rank]["fixed_cpu_calls"][start, finish]
                cuts = self.progress_cuts.get((rank, start, finish), ())
                if any(cut.service_ns != duration for cut in cuts):
                    self.gaps.setdefault("unqualified_work_service_partitions", []).append((rank, start, finish))
                basis = "observed_cpu_protocol_occupancy" if duration is not None else "missing_cpu_protocol_occupancy"
                yield ServiceInterval(rank, None, start, finish, "host", duration, basis, cuts, call, operation)

    def collective_window(self, rank, graph, piece, captured):
        start, finish = piece["start"], piece["finish"]
        destination = rank, finish
        if destination not in self.peers:
            return None

        posts = self.peers[destination] | {(rank, start)}
        lane = piece["resource"]
        if graph["resources"][lane]["kind"] == "thread":
            ports = {port["name"]: port["event"] for port in graph["scopes"][piece["scope"]]["ports"]}
            if (ports.get("call.enter"), ports.get("call.return")) != (start, finish):
                return None

            # Host participants share the RAW clock. Require real call boundaries for the full rendezvous.
            if any(type(self.clock.point(*event)) is not int for event in posts | {destination}):
                return None
        else:
            events = graph["evidence"]["events"]
            if (lane, events[start]["order"], events[finish]["order"]) not in captured:
                return None

        readiness = [self.clock.timestamp(*event) for event in posts]
        completion = self.clock.timestamp(*destination)
        return None if completion is None or None in readiness else (max(readiness), completion)

    def work_fields(self, interval):
        if isinstance(interval, CpuTransportInterval):
            descriptor = interval.post.descriptor
            return {"kind": interval.kind, "scope": "cpu_protocol", "site": CpuProtocolWork.readiness_operation(descriptor),
                    "phase": "", "direction": "receive" if descriptor["receiving"] else "send", "caller": None}
        if call := interval.cpu_call:
            return {"kind": interval.kind, "scope": "cpu_protocol", "site": interval.cpu_operation,
                    "phase": "", "direction": "receive" if call.descriptor["receiving"] else "send", "caller": None}
        graph = self.case.sources[interval.rank][0].graph
        piece = graph["work"][interval.index]
        scope = graph["scopes"][piece["scope"]]
        copy = piece.get("copy", {})
        return {"kind": interval.kind, "scope": scope["role"], "site": scope["site"], "phase": scope["work"].get("phase"),
                "direction": copy.get("direction"), "caller": copy.get("caller")}

    def detail(self, interval):
        if isinstance(interval, CpuTransportInterval):
            call = interval.post
            return {"rank": call.returned[0], **self.work_fields(interval), "message": call.key,
                    "consumer": interval.consumer, "ready_window_ns": interval.window_ns}
        return {"rank": interval.rank, "work_index": interval.index, **self.work_fields(interval),
                "start": interval.start, "finish": interval.finish}

    def external_cost(self, interval):
        if isinstance(interval, CpuTransportInterval):
            call = interval.post
            operation = CpuProtocolWork.readiness_operation(call.descriptor)
            execution = interval.execution
        elif call := interval.cpu_call:
            operation, execution = interval.cpu_operation, call.execution
        else:
            return None
        identity, features = CpuProtocolWork.service(call.descriptor, execution, operation)
        return call.returned[0], identity, features

    def entry_releases(self):
        causal = self.case.graph

        # Prefix protocol events can constrain serving readiness without turning startup into modeled work.
        has_predecessor = {target for source, targets in causal.next.items() if source not in self.external_cpu_points
                           for target in targets}
        entries = {point: max(0, self.case.cpu.native_times[point] - self.origin_ns) for point in self.external_cpu_points}
        for rank, event in causal.positions:
            if event < 0 or (rank, event) in has_predecessor or (rank, event) in self.signal_submissions:
                continue
            if (rank, event) in self.elided_waits:
                continue
            graph = self.case.sources[rank][0].graph
            resource = graph["resources"][graph["events"][event]["resource"]]
            seen = self.case.clocks[rank].window(event)
            if resource["kind"] != "thread" or seen is None or seen[0] != seen[1] or seen[0] < self.origin_ns:
                self.gaps.setdefault("unqualified_entry_boundaries", []).append({"rank": rank, "event": event})
                continue

            # A measured host entry releases the fixed graph; internal boundaries follow simulated dependencies.
            entries[rank, event] = seen[0] - self.origin_ns
        return entries

    def timeline(self):
        graph = self.case.graph
        timeline = FixedTimeline(self)
        indices = timeline.indices
        fixed = {(indices[source], indices[target]): sum(conditions.values())
                 for (source, target), conditions in self.host_conditions.items()}
        order_cuts = self.progress_cuts.copy()

        for interval in self.intervals:
            if isinstance(interval, CpuTransportInterval):
                timeline.transport(interval)
                continue
            rank, begin, end = interval.rank, interval.start, interval.finish
            start, finish = indices[rank, begin], indices[rank, end]
            delay = fixed.pop((start, finish), 0)
            order_cuts.pop((rank, begin, end), None)
            if interval.kind == "collective":
                start = timeline.event()
                readiness = self.peers[rank, end] | {(rank, begin)}
                timeline.dependencies.extend((indices[predecessor], start, 0) for predecessor in readiness)
            timeline.path(rank, start, finish, delay, interval.cuts, modeled=True)

        # Fences and device dispatches add measured handling after their causal frontier.
        gates = {}
        for target, tail in (self.device_conditions | self.host_fences).items():
            gate = timeline.event()
            timeline.dependencies.append((gate, indices[target], tail))
            gates[target] = gate
        for target, source in self.signal_submissions.items():
            timeline.dependencies.append((indices[source], gates[target], 0))

        # Unpublished records use their measured submission and stream predecessor on the fixed timeline.
        for target, (previous, records) in self.device_records.items():
            rank, _ = target
            predecessor = indices[rank, previous]
            for record in records:
                post = timeline.progress[rank, record.post] if isinstance(record.post, WorkProgress) else indices[rank, record.post]
                complete = timeline.event()
                timeline.dependencies.extend(((post, complete, record.duration_ns),
                                              (predecessor, complete, record.duration_ns)))
                predecessor = complete
            timeline.dependencies.append((predecessor, gates[target], 0))

        # Fixed delays at regular dependencies also include measured handoffs between distinct host threads.
        for source, targets in graph.next.items():
            for target in targets:
                pair = indices[source], indices[target]
                delay = fixed.pop(pair, 0)
                cuts = order_cuts.pop((source[0], source[1], target[1]), ()) if source[0] == target[0] else ()
                timeline.path(source[0], pair[0], gates.get(target, pair[1]), delay, cuts, modeled=False)
        if fixed or order_cuts:
            raise ValueError("fixed timing condition does not belong to a verified dependency")
        self.protocol_positions = timeline.protocol
        self.input_positions = {cut: (reached, timeline.progress[cut.rank, cut]) for cut, reached in timeline.input_progress.items()}
        return timeline.document()

    def compare_active_input_reads(self, replayed, *, measured=False):
        differences = []
        waiting_ns, delayed = 0, 0
        for cut, (progress, delivered) in self.input_positions.items():
            observed = cut.observed_ns - self.origin_ns
            reached, actual = replayed[progress], replayed[delivered]
            waiting_ns += actual - reached
            delayed += actual > reached
            if actual != max(observed, reached) or (measured and reached != observed):
                differences.append({"rank": cut.rank, "requests": self.active_input_reads[cut],
                                    "observed_ns": observed, "local_progress_ns": reached, "replayed_ns": actual})
        return {"checked": len(self.input_positions), "delayed_deliveries": delayed, "total_delivery_wait_ns": waiting_ns,
                "different": len(differences), "differences": differences}

    def transport_baseline(self, replayed):
        report = {"checked": len(self.cpu_transfers), "initial_conditions": len(self.cpu_initial_readiness), "different": 0}
        for interval in self.cpu_transfers:
            stamp = self.origin_ns + replayed[self.protocol_positions[interval.post.posted]]
            begin, end = interval.window_ns
            if not begin <= stamp <= end:
                if not report["different"]:
                    report["first_difference"] = {**self.detail(interval), "replayed_raw_ns": stamp}
                report["different"] += 1
        return report

    def replay_services(self, durations):
        steps = []
        for interval, duration in zip(self.intervals, durations, strict=True):
            previous = 0
            for cut in interval.cuts:

                # Fixed scenarios preserve the measured progress at each intra-work boundary.
                prefix = duration * cut.service_prefix_ns // interval.service_ns
                steps.append(prefix - previous)
                previous = prefix
            steps.append(duration - previous)
        return steps

    def completion_windows(self, predecessors):
        graph = self.case.graph
        windows = {}
        for rank, event in graph.positions:
            key = rank, event
            if event >= 0:
                observed = self.case.clocks[rank].window(event)
            else:
                stamp = self.case.cpu.native_times[key]
                observed = stamp, stamp
            if key not in self.observer_joins:
                windows[key] = observed
                continue

            # At a logical join the latest original prerequisite makes the result ready.
            parents = self.readiness_parents(key, predecessors, self.elided_waits)
            sources = [windows[parent] for parent in parents]
            if observed is None or not sources or None in sources:
                self.gaps.setdefault("unqualified_observer_joins", []).append(key)
                windows[key] = None
                continue
            first = max(begin for begin, _ in sources)
            last = min(observed[1], max(end for _, end in sources))
            if first > last:
                raise ValueError("observer marker precedes a verified join prerequisite")
            windows[key] = first, last
        return list(windows.values())

    def check_baseline(self, replayed):
        checked, max_clock_width = 0, 0
        evidence, differences = Counter(), Counter()
        first = None
        for (rank, event), position in self.case.graph.positions.items():
            if (rank, event) in self.external_cpu_points:
                evidence["external_protocol_condition"] += 1
                continue
            bounds = self.event_windows[position]
            if bounds is None:
                evidence["unmeasured"] += 1
                continue
            checked += 1
            time = replayed[position]
            begin, end = (stamp - self.origin_ns for stamp in bounds)
            if (rank, event) in self.observer_joins:
                evidence["derived_join"] += 1
            elif event < 0:
                evidence["host_protocol_raw"] += 1
            elif self.case.clocks[rank].device[event]:
                evidence["calibrated_device"] += 1
                max_clock_width = max(max_clock_width, end - begin)
            else:
                evidence["host_raw"] += 1
            if begin <= time <= end:
                continue
            differences["early" if time < begin else "late"] += 1
            if first is None or begin < first["measured_window_ns"][0]:
                call = self.cpu_points.get((rank, event))
                scope = {"cpu_protocol": call.source()} if call else self.case.sources[rank][0].graph["events"][event]["scope"]
                first = {"rank": rank, "event": event, "scope": scope,
                         "measured_window_ns": [begin, end], "replayed_ns": time}
        return {"checked_boundaries": checked, "observer_joins": len(self.observer_joins),
                "boundary_evidence": dict(evidence), "max_device_clock_window_ns": max_clock_width,
                "differences": dict(differences), "first_difference": first}

    def local_unexplained_waits(self, plan, services):
        original = len(self.event_windows)
        latest = [None if window is None else window[1] - self.origin_ns for window in self.event_windows]
        latest.extend(plan["releases_ns"][original:])
        work = ((source, target, lag + service)
                for (source, target, lag), service in zip(plan["work"], services, strict=True))
        incoming = plan["releases_ns"].copy()
        unknown = set()
        outgoing, remaining = defaultdict(list), Counter()
        for source, target, lag in chain(plan["dependencies"], work):
            outgoing[source].append((target, lag))
            if target >= original:
                remaining[target] += 1

        # Original boundaries are fixed witnesses here. Only auxiliary cuts need causal propagation.
        ready = list(range(original))
        ready.extend(index for index in range(original, len(latest)) if not remaining[index])
        for source in ready:
            stamp = latest[source]
            for target, lag in outgoing[source]:
                if target < original:
                    if stamp is None:
                        unknown.add(target)
                    else:
                        incoming[target] = max(incoming[target], stamp + lag)
                    continue
                if latest[target] is not None:
                    latest[target] = None if stamp is None else max(latest[target], stamp + lag)
                remaining[target] -= 1
                if not remaining[target]:
                    ready.append(target)

        # Use original predecessor evidence here so earlier replay drift cannot create a local gap.
        checked, groups = 0, {}
        for (rank, event), position in self.case.graph.positions.items():
            window = self.event_windows[position]
            if window is None or position in unknown or (rank, event) in self.external_cpu_points:
                continue
            checked += 1
            gap = window[0] - self.origin_ns - incoming[position]
            if gap <= 0:
                continue
            if event < 0:
                role = "cpu_protocol"
            else:
                graph = self.case.sources[rank][0].graph
                resource = graph["resources"][graph["events"][event]["resource"]]
                role = "observer_join" if (rank, event) in self.observer_joins else resource["role"]
            group = groups.setdefault(role, {"count": 0, "max_lower_ns": 0})
            group["count"] += 1
            if gap > group["max_lower_ns"]:
                group.update(max_lower_ns=gap, rank=rank, event=event)
        return {"checked_boundaries": checked, "by_role": groups}
