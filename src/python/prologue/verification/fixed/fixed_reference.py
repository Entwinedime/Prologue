"""Bind same-run service evidence and request boundaries to observed execution."""

from collections import Counter, defaultdict
from dataclasses import replace
from itertools import chain

from prologue.verification.core import FixedPublications
from prologue.verification.core.ingress import IngressEvidence
from prologue.verification.core.source_clock import ClockRealization, CounterPoint
from prologue.verification.fixed.fixed_timeline import (
    CpuTransportInterval,
    FixedDeviceRecord,
)

from .fixed_boundaries import FixedBoundaries, readiness_parents
from .fixed_context import CpuReadinessWake, ExternalWake, KernelWake, LocalCallWake, WorkProgress
from .fixed_services import FixedServices


class FixedReference:
    """Use the original graph as the fixed input and keep modeled service time separate from recorded boundary
    evidence.
    """

    def __init__(self, case):
        self.case = case
        self.peers = {}
        for source, target in case.connections.communication.native_communication_edges:
            self.peers.setdefault(target, set()).add(source)

        self.gaps = {}
        self.input_releases = {}
        self.active_input_reads = {}
        self.requests = self.align_requests()
        self.cpu_calls = [
            call for calls in case.cpu.native_calls.values() for call in calls if call.execution["in_flow"]
        ]
        self.cpu_points = {
            point: call
            for call in self.cpu_calls
            for point in (call.entered, call.posted, call.returned)
            if point is not None
        }
        self.external_cpu_points = case.cpu.native_times.keys() - self.cpu_points.keys()
        self.specialization_loads = {
            rank: raw.get("specialization_loads") for rank, (_, raw, _) in case.sources.items()
        }

        dispatches = {
            (rank, start): dispatch
            for rank, record in case.measurements.items()
            for start, dispatch in record["device_releases"].items()
        }
        self.clock = ClockRealization(case.clocks, case.graph, case.cpu.native_times)

        for target, dispatch in dispatches.items():
            submitted = target[0], dispatch.submitted

            # A native signal's driver submission precedes the physical command entry.
            if dispatch.submitted_at_completion and target not in case.graph.next[submitted]:
                self.clock.constrain_order(submitted, target)

        self.clock.solve()
        self.host_waits = dict(self.input_releases)
        self.host_waits.update(
            ((rank, wait.stopped), wait)
            for rank, record in case.measurements.items()
            for wait in record["fixed_host_waits"]
        )

        # Match measured protocol readiness before assigning the remaining wakes to their producer lanes.
        self.cpu_initial_readiness = {}
        self.cpu_transfers = self.bind_cpu_transfers()
        self.bind_thread_wakes()

        self.bind_graph_conditions(dispatches)
        self.progress_cuts = self.collect_progress_cuts()
        self.services = FixedServices(case, self.clock, self.peers, self.progress_cuts, self.cpu_calls, self.gaps)
        self.intervals = [*self.services.services(), *self.cpu_transfers]
        self.entries = self.entry_releases()
        unbound = {
            rank: record["unbound_queue_notifications"]
            for rank, record in case.measurements.items()
            if record["unbound_queue_notifications"]
        }

        if unbound:
            self.gaps["unbound_queue_notifications"] = unbound

    def bind_graph_conditions(self, dispatches):
        case = self.case
        self.observer_joins = {
            (rank, event) for rank, record in case.measurements.items() for event in record["observer_joins"]
        }
        self.elided_waits = {(rank, event) for rank, events in case.logical.items() for event in events}
        fences = [
            (rank, source, target)
            for rank, record in case.measurements.items()
            for source, target in sorted(record["host_fences"])
        ]
        signals = [
            (rank, start, row.completion)
            for (rank, start), row in dispatches.items()
            if row.completion is not None and (rank, row.completion) not in self.observer_joins
        ]
        completions = {(rank, target) for rank, _, target in chain(fences, signals)}
        joins = self.observer_joins | self.elided_waits
        predecessors = self.predecessors(joins | dispatches.keys() | completions)

        self.boundaries = FixedBoundaries(
            case, self.origin_ns, self.observer_joins, self.elided_waits, self.cpu_points,
            self.external_cpu_points, predecessors, self.gaps,
        )
        self.publications = FixedPublications(case, self.requests, self.origin_ns)

        if missing := self.publications.coverage_gaps():
            self.gaps["incomplete_publication_boundaries"] = missing
        self.host_conditions = {
            ((rank, source), (rank, target)): conditions
            for rank, record in case.measurements.items()
            for (source, target), conditions in record["fixed_host_delays"].items()
        }

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

    def bind_thread_wakes(self):
        owners = {
            kernel: (rank, thread)
            for rank, record in self.case.measurements.items()
            for thread, kernel in record["kernel_threads"].items()
        }
        self.cross_worker_wakes = 0
        self.post_graph_host_wakes = 0

        for (rank, stopped), wait in tuple(self.host_waits.items()):
            witness = wait.source
            if not isinstance(witness, KernelWake):
                continue

            location = (
                (rank, witness.local_thread) if witness.local_thread is not None else owners.get(witness.kernel_thread)
            )

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

            if anchor is None or (
                isinstance(anchor, (WorkProgress, LocalCallWake))
                and (owner, anchor.finish) not in self.case.graph.next[owner, anchor.start]
            ):
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
        posts = {
            (call.key, rank): call
            for rank, calls in case.cpu.native_calls.items()
            for call in calls
            if call.phase == "post"
        }
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
                matches = [
                    wait
                    for wait in sleeps
                    if isinstance(wait.source, KernelWake)
                    and wait.source.kernel_thread == kernel
                    and kernel is not None
                    and first <= wait.source.raw_ns <= last
                ]
                ready = first

                if matches:
                    ready = min(wait.source.raw_ns for wait in matches)
                    for wait in matches:
                        delay = wait.source.raw_ns - ready + wait.delay_ns
                        self.host_waits[rank, wait.stopped] = replace(
                            wait, source=CpuReadinessWake(call.posted[1]), delay_ns=delay
                        )
                elif sleeps and last >= case.cpu.native_times[self.cpu_points[consumer].entered]:
                    unbound.append((key, rank))

                # An already posted message contributes initial readiness without charging pre-window transfer work.
                if posted < self.origin_ns:
                    if consumer in self.cpu_points:
                        self.cpu_initial_readiness[call.posted] = consumer, ready
                else:
                    execution = call.execution if call.execution["in_flow"] else active[0]
                    transfers.append(
                        CpuTransportInterval(call, consumer, frontiers, execution, (first, last), ready - posted)
                    )

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
            parents = readiness_parents(returned, predecessors, self.elided_waits)
            if entry not in parents:
                unqualified.append((rank, begin, end))
                continue

            entered = self.clock.timestamp(*entry)
            finish = self.clock.timestamp(*returned)
            readiness = {parent: self.boundaries.windows[positions[parent]] for parent in parents}
            if any(
                window is None or parent in self.observer_joins and window[1] > entered
                for parent, window in readiness.items()
            ):
                unqualified.append((rank, begin, end))
                continue

            # A verified completion follows the actual producer and the wait entry on either lane.
            ready = max(
                self.clock.timestamp(*parent) + sum(self.host_conditions.get((parent, returned), {}).values())
                for parent in parents
                if parent not in self.observer_joins
            )

            if ready > finish:
                unqualified.append((rank, begin, end))
                continue

            qualified[returned] = finish - ready
            counts["frontier_in_call" if ready > entered else "ready_on_entry"] += 1
            counts["return_delay_ns"] += finish - ready

        if unqualified:
            self.gaps[gap] = {"count": len(unqualified), "first": unqualified[0]}
        return qualified, {"qualified": len(qualified), **counts}

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
        windows = [self.boundaries.windows[positions[event]] for event in frontier]
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
            window = self.boundaries.windows[positions[parent]]

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
            parents = readiness_parents(target, predecessors, joins)
            previous = rank, dispatch.previous
            preceding = dispatch.preceding_window_ns
            posted = self.boundaries.windows[positions[source]]
            observed_start = self.boundaries.windows[positions[target]]

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
                basis = (
                    "preceding_task_before_calibration" if preceding[0] is None else "preceding_task_already_complete"
                )
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
        (self.origin_ns,) = {entry.origin_ns for entry in case.submissions}
        intake = []

        for instance, submitted in zip(case.manifest["instances"], case.submissions, strict=True):
            if instance["mode"] != "decode":
                intake.append(
                    [
                        {**row, "arrival_ns": submitted.by_request[name]["submitted_ns"]}
                        for name, row in submitted.requests.items()
                    ]
                )

            self.bind_ingress(instance, submitted)

        (workload,) = intake
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

                submission = self.origin_ns + max(
                    submitted.by_request[name]["submitted_ns"] for name in read["requests"]
                )
                attempt = reads.get(received)

                if (
                    attempt is not None
                    and attempt["context"] is not None
                    and not IngressEvidence.idle(attempt["context"])
                    and submission <= attempt["begin_raw_ns"]
                ):
                    cut = host["fixed_host_context"].input_read_progress(attempt["begin_raw_ns"], received)
                    if cut is not None:
                        self.active_input_reads[cut] = read["requests"]
                        continue

                self.gaps.setdefault("unqualified_input_readiness", []).append(
                    {
                        "rank": rank,
                        "requests": read["requests"],
                        "submission_raw_ns": submission,
                        "received_raw_ns": received,
                    }
                )

    def entry_releases(self):
        causal = self.case.graph

        # Prefix protocol events can constrain serving readiness without turning startup into modeled work.
        has_predecessor = {
            target
            for source, targets in causal.next.items()
            if source not in self.external_cpu_points
            for target in targets
        }
        entries = {
            point: max(0, self.case.cpu.native_times[point] - self.origin_ns) for point in self.external_cpu_points
        }

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
