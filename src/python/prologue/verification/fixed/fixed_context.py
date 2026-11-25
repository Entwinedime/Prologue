"""Project independently observed host conditions onto serial steps of a fixed graph."""

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from prologue.common.observation import ThreadService
from prologue.verification.core.adapters import platform_evidence
from prologue.verification.core.protocol import IngressEvidence

from .host_queues import HostQueueEvidence


@dataclass(frozen=True, slots=True)
class HostStep:
    start: int
    finish: int
    begin_ns: int
    end_ns: int
    modeled: bool
    service_ns: int | None


@dataclass(frozen=True, slots=True)
class WorkProgress:
    rank: int
    start: int
    finish: int
    observed_ns: int
    service_prefix_ns: int
    fixed_prefix_ns: int
    service_ns: int


@dataclass(frozen=True, slots=True)
class LocalCallWake:
    rank: int
    start: int
    finish: int
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class ExternalWake:
    raw_ns: int


@dataclass(frozen=True, slots=True)
class CpuReadinessWake:
    post: int


@dataclass(frozen=True, slots=True)
class KernelWake:
    raw_ns: int
    local_thread: int | None
    kernel_thread: int | None = None


@dataclass(frozen=True, slots=True)
class HostWait:
    stopped: WorkProgress
    source: WorkProgress | LocalCallWake | ExternalWake | KernelWake | CpuReadinessWake | tuple[int, int] | None
    delay_ns: int
    take_entry: WorkProgress | None = None


class FixedThreadActivity(ThreadService):

    # Linux futex command and options recorded by the syscall tracepoint.
    futex_flags = 128 | 256
    futex_wait_ops = frozenset({0, 9})
    interrupt_flags = 0x08 | 0x10 | 0x40

    def __init__(self, rows, dequeues, excluded, runtime):
        super().__init__(rows, dequeues, excluded)
        covered = [row for row in rows if row["thread"] in self.activity]
        self.targeted = {row["thread"] for row in covered if "target_wakes" in row}
        self.calls = platform_evidence().thread_intervals([
            platform_evidence().interval(row["begin_raw_ns"], row["end_raw_ns"], row["thread"], row)
            for row in runtime if row["result"] == 0 and row["thread"] in self.targeted
        ])
        self.runnable, self.blocked_at, self.sleep = {}, {}, {}
        self.kernel_threads = {row["thread"]: row["kernel_thread"] for row in rows if "kernel_thread" in row}
        self.wakes = defaultdict(list)
        self.futex_waits = defaultdict(list)
        for row in covered:
            thread = row["thread"]
            runnable, blocked = self.switch_points(row)
            self.runnable[thread], self.blocked_at[thread] = runnable, blocked
            pauses = set(runnable)
            self.sleep[thread] = [
                (out, resume) for (out, running), (resume, resumed) in pairwise(row["switches"])
                if not running and resumed and (out, resume) not in pauses
            ]
            pids, entered = set(), None
            for kind, stamp, *payload in row.get("kernel_events", ()):
                if kind == "switch_out":
                    pids.add(payload[0])
                elif kind == "wake":
                    self.wakes[payload[0]].append(KernelWake(stamp, thread))
                elif kind == "futex_enter":
                    operation = payload[1] & ~self.futex_flags
                    entered = stamp if operation in self.futex_wait_ops else None
                elif kind == "futex_exit":
                    if entered is not None and payload[0] == 0:
                        self.futex_waits[thread].append((entered, stamp))
                    entered = None
            if thread in self.kernel_threads and pids - {self.kernel_threads[thread]}:
                raise ValueError("worker activity changes its kernel thread identity")
            if len(pids) == 1:
                self.kernel_threads[thread], = pids

        local_threads = {kernel: local for local, kernel in self.kernel_threads.items()}
        if len(local_threads) != len(self.kernel_threads):
            raise ValueError("native worker repeats a kernel thread identity")
        for row in covered:
            if "target_wakes" not in row:
                continue
            recipient = self.kernel_threads[row["thread"]]
            for stamp, producer, flags in row["target_wakes"]:

                # Interrupt tracepoints carry the interrupted task's PID, not a user-space waker.
                if flags & self.interrupt_flags:
                    self.wakes[recipient].append(KernelWake(stamp, None))
                else:
                    self.wakes[recipient].append(KernelWake(stamp, local_threads.get(producer), producer))
        for stamps in self.wakes.values():
            stamps.sort(key=lambda wake: wake.raw_ns)
        self.report = {
            "threads": sum("kernel_events" in row for row in covered),
            "target_wakes": sum(len(row.get("target_wakes", ())) for row in covered),
            "additional_runnable_pauses": sum(
                len(self.runnable[row["thread"]]) - len(row["preemptions"]) for row in covered
            ),
        }

    @staticmethod
    def switch_points(row):
        pauses = {tuple(pair) for pair in row["preemptions"]}
        blocked = {}
        states = [
            (stamp, payload[-1]) for kind, stamp, *payload in row.get("kernel_events", ()) if kind == "switch_out"
        ]
        stamps = [stamp for stamp, _ in states]
        on_cpu = row["window_ns"][0]
        for (out, running), (resume, _) in pairwise(row["switches"]):
            if running:
                on_cpu = out
                continue

            # The matching scheduler sample belongs to this run slice on the same per-thread perf buffer.
            first = bisect_left(stamps, on_cpu)
            last = bisect_right(stamps, out, lo=first)
            if last == first + 1:
                stamp, state = states[first]
                if state == 0:
                    pauses.add((out, resume))
                else:
                    blocked[out] = on_cpu, stamp
        return sorted(pauses), blocked

    def waking(self, thread, begin, end):
        stamps = self.wakes.get(self.kernel_threads.get(thread), ())
        first = bisect_left(stamps, begin, key=lambda wake: wake.raw_ns)
        last = bisect_right(stamps, end, lo=first, key=lambda wake: wake.raw_ns)
        return stamps[first:last]

    def sleep_wakes(self, thread, out, resumed):
        for entered, returned in self.within(self.futex_waits.get(thread, ()), out, resumed):
            if entered <= out < resumed <= returned:

                # Perf can record the switch-out later than the kernel wake inside the same syscall.
                return self.waking(thread, entered, resumed)
        transition = self.blocked_at[thread].get(out)
        if transition is None:
            return self.waking(thread, out, resumed)
        entered_cpu, blocked = transition
        wakes = self.waking(thread, blocked, resumed)
        if wakes:
            return wakes

        opened, closed, _ = self.activity[thread]
        if thread in self.targeted and opened < entered_cpu:

            # A targeted wake can hit the blocked task before sched_switch records it leaving the CPU.
            early = self.waking(thread, entered_cpu, blocked)
            if len(early) == 1:
                return early

        # A native call independently bounds a single sleep even if earlier wakes share the CPU slice.
        call = self.calls.enclosing(platform_evidence().interval(out, resumed, thread, {}))
        if call is None or not opened <= call.begin < call.end <= closed:
            return ()
        if self.within(self.sleep[thread], call.begin, call.end) != [(out, resumed)]:
            return ()
        witnesses = self.waking(thread, call.begin, call.end)
        return witnesses if len(witnesses) == 1 and witnesses[0].raw_ns <= blocked else ()


class FixedHostContext:
    @classmethod
    def retain(
        cls, native_case, rank, facts, graph, native_evidence, host_fences, launches, payload_times, report, details
    ):
        records = {key: facts[key] for key in ("resources", "environment", "measurements")}
        # Timed work and published physical signals retain their actual measured endpoints.
        timed = {piece[side] for piece in graph["work"] for side in ("start", "finish")}
        timed.update(port["event"] for scope in graph["scopes"] for port in scope["ports"]
                     if port["use"] == "signal" and port["signal"].startswith("event/"))
        records["observer_joins"] = native_evidence.observer_joins - timed
        releases = native_evidence.sdk.release_evidence(
            graph, launches, native_evidence.queues, native_case.clocks[rank], records["observer_joins"]
        )
        omitted = {record for release in releases.values() for record in release.omitted}
        owner = native_case.ranks[rank]
        deliveries = zip(native_case.manifest["instances"], native_case.submissions, strict=True)
        delivery = next(submitted for instance, submitted in deliveries if instance["id"] == owner)
        context = cls(native_case.capture, rank, native_evidence, delivery)
        calls = native_case.cpu.native_calls[rank]
        measured = context.project(
            native_evidence, graph, host_fences, omitted, details, calls, native_case.cpu.native_times
        )
        delays, waits, idle_inputs, posts, queues, protocol = measured
        records["fixed_cpu_calls"] = protocol
        report["native_host_queues"] = queues
        records["fixed_host_delays"] = delays
        records["fixed_idle_inputs"] = idle_inputs
        records["host_fences"] = host_fences
        records["fixed_driver_posts"] = posts
        records["device_releases"] = releases
        records["device_payload_times"] = payload_times
        records["fixed_host_waits"] = waits
        records["fixed_host_context"] = context
        records["kernel_threads"] = context.active.kernel_threads
        records["unbound_queue_notifications"] = (
            queues["queue_completion_notifications"] - queues["bound_to_fixed_work"]
        )

        native_case.measurements[rank] = records

    def __init__(self, capture, rank, evidence, delivery):
        self.rank = rank
        self.native = evidence.native
        self.delivery = delivery
        self.control_thread = evidence.window.thread
        activity = evidence.native["host_observation"].get("activity")
        rows = []
        if activity is not None:
            if activity != f"rank-{rank}-thread-activity.jsonl":
                raise ValueError("fixed host activity file belongs to a different worker")
            with (capture / activity).open() as source:
                rows = [json.loads(line) for line in source]

        # Waits triggered by work on other resources still follow their original dependencies.
        observer, dequeues, waits = evidence.replay_host_intervals()
        self.active = FixedThreadActivity(rows, dequeues, [*observer, *waits], evidence.runtime)
        self.observer = ThreadService.by_thread(observer)

    def interruptions(self, tid, begin, end):
        activity = [(max(begin, first), min(end, last))
                    for first, last in self.active.within(self.observer.get(tid, ()), begin, end)]
        overhead = sum(last - first for first, last in activity)

        # Explicit observation intervals include their own interruptions. Business pauses are counted once.
        pauses = [(first, last) for first, last in self.active.within(self.active.runnable.get(tid, ()), begin, end)
                  if begin <= first <= last <= end]
        paused = sum(last - first for first, last in pauses) - sum(
            last - first for first, last in self.active.intersect(pauses, activity))
        return overhead, paused

    def idle_waits(self, graph, events, work, notifications):
        if not self.native["request_ingress"] or IngressEvidence.attempts(self.native)["status"] == "unobserved":
            return {}

        thread = self.control_thread
        reads = self.native["request_input_attempts"]
        sleepers = self.active.sleep.get(thread, ())
        completions = {tuple(row["off_cpu_ns"]) for row in notifications}
        excluded = self.active.excluded.get(thread, ())
        window_start, window_end = graph["evidence"]["window_ns"]
        submitted_ns = {
            native: self.delivery.origin_ns + self.delivery.by_request[request]["submitted_ns"]
            for native, request in self.delivery.aliases.items()
        }
        cuts = {}

        for prior, received in pairwise(reads):
            read_begin, read_end = received["begin_raw_ns"], received["received_raw_ns"]
            if not window_start < read_end <= window_end or prior["message_count"]:
                continue
            if not IngressEvidence.idle(received["context"]):
                continue
            if received["message_count"] and not received["requests"]:
                continue

            prior_end = prior["received_raw_ns"]
            stamps = [submitted_ns[request] for request in received["requests"]]
            candidates = [
                (stopped, resumed) for stopped, resumed in self.active.within(sleepers, prior_end, read_begin)
                if prior_end <= stopped < resumed <= read_begin
                and all(stopped <= stamp <= resumed for stamp in stamps)
                and (stopped, resumed) not in completions
                and not self.active.within(excluded, stopped, resumed)
            ]
            if len(candidates) != 1:
                continue
            (stopped, resumed), = candidates

            # The receiver can execute other idle steps between resuming and reading the request.
            slot = bisect_right(events, stopped, key=lambda event: event[1])
            if not 0 < slot < len(events):
                continue
            _, _, source = events[slot - 1]
            _, end, target = events[slot]
            if resumed > end or (source, target) in work:
                continue
            scope = graph["scopes"][graph["events"][source]["scope"]]
            if not IngressEvidence.idle(scope["work"].get("context")):
                continue

            cuts[source, target, stopped, resumed] = read_end, bool(received["requests"])

        if not cuts:
            return cuts
        edges = {cut[:2] for cut in cuts}
        ordered = {(edge["source"], edge["target"]) for edge in graph["edges"]
                   if edge["kind"] == "order" and (edge["source"], edge["target"]) in edges}
        return {cut: read for cut, read in cuts.items() if cut[:2] in ordered}

    def host_time(self, thread, begin, end, *, modeled, submission):
        if begin == end:
            return 0, {}

        occupied = self.active.duration(thread, begin, end, False)
        overhead, paused = self.interruptions(thread, begin, end)
        service = occupied if modeled else 0

        # Submission service uses dequeue occupancy. Preserve other observed CPU activity as a fixed condition.
        if modeled and submission:
            service = self.active.duration(thread, begin, end, True)
        outside = None if occupied is None or service is None else occupied - service
        conditions = {"observer_ns": overhead, "runnable_pause_ns": paused, "outside_work_ns": outside}
        return None if occupied is None else service, {name: value for name, value in conditions.items() if value}

    def progress(self, thread, step, stamp):
        if step.service_ns is None or (step.modeled and not step.service_ns):
            return None

        service, conditions = self.host_time(thread, step.begin_ns, stamp, modeled=step.modeled,
                                             submission=thread in self.submissions)
        if service is None:
            return None
        return WorkProgress(
            self.rank, step.start, step.finish, stamp, service, sum(conditions.values()), step.service_ns
        )

    def step_at(self, thread, stamp):
        host = self.steps.get(thread, ())
        position = bisect_right(host, stamp, key=lambda step: step.begin_ns) - 1
        if position >= 0 and stamp < host[position].end_ns:
            return host[position]
        return None

    def progress_at(self, thread, stamp):
        if step := self.step_at(thread, stamp):
            return self.progress(thread, step, stamp)
        return None

    def input_read_progress(self, begin, end):
        thread = self.control_thread
        step = self.step_at(thread, begin)
        if step is None or step.modeled or end > step.end_ns:
            return None

        running = self.active.running_intervals(thread, begin, end)
        if running is None:
            return None
        segments = list(running)
        if not segments or segments[0][0] != begin or segments[-1][1] != end:
            return None

        # Runnable preemptions stay in the fixed host step. A blocked pull needs its own readiness evidence.
        pauses = self.active.within(self.active.runnable[thread], begin, end)
        if pauses != [(left[1], right[0]) for left, right in pairwise(segments)]:
            return None
        return self.progress(thread, step, end)

    def synchronous_calls(self, graph, threads):
        windows = []
        events = graph["evidence"]["events"]
        for piece in graph["work"]:
            lane = piece["resource"]
            thread = threads.get(lane)
            if thread is not None and piece["kind"] == "collective":
                begin, end = (events[event]["time_ns"] for event in (piece["start"], piece["finish"]))
                if begin is not None and end is not None and begin < end:
                    windows.append(platform_evidence().interval(begin, end, thread, piece))
                continue

            resource = graph["resources"][lane]
            if resource["kind"] != "service":
                continue
            scope = graph["scopes"][piece["scope"]]
            origin, = scope["bindings"]["source_scopes"]
            native = self.native["scopes"][origin]
            if "service" in native:
                interval = native["host_interval"]
                windows.append(
                    platform_evidence().interval(
                        interval["begin_raw_ns"], interval["end_raw_ns"], native["thread"], piece
                    )
                )
        return platform_evidence().thread_intervals(windows)

    def call_at(self, thread, stamp):
        call = self.call_windows.enclosing(platform_evidence().interval(stamp, stamp, thread, {}))
        return call if call is not None and stamp < call.end else None

    def serial_wake_anchor(self, thread, stamp):
        if thread not in self.thread_bounds:
            return ExternalWake(stamp), 0
        first, last, event = self.thread_bounds[thread]
        if first is None or last is None:
            return None, 0
        if stamp < first:
            return ExternalWake(stamp), 0

        # The graph owns the last event. A measured wake later on this serial thread follows that event.
        if stamp >= last:
            return (self.rank, event), stamp - last
        return self.progress_at(thread, stamp), 0

    def kernel_wait(self, thread, blocked, stopped, *, read_wake):
        out, resumed = blocked
        if self.active.within(self.active.excluded.get(thread, ()), out, resumed):
            return None
        wake = read_wake
        if wake is None:
            witnesses = self.active.sleep_wakes(thread, out, resumed)
            if len(witnesses) != 1:
                return None
            wake, = witnesses

        kernel = self.active.kernel_threads.get(wake.local_thread, wake.kernel_thread)
        witness = KernelWake(wake.raw_ns, wake.local_thread, kernel)
        return HostWait(stopped, witness, resumed - wake.raw_ns)

    def queue_wait(self, notice, stopped, positions, queues):
        producer = notice["producer"]
        taken, completed = notice["dequeue_started_ns"], notice["dequeue_finished_ns"]
        resumed = notice["off_cpu_ns"][1]
        endpoint = positions.get((producer, notice["dequeue_order"]))
        if endpoint is not None:
            return HostWait(stopped, (self.rank, endpoint), resumed - completed)

        arrival = queues.unowned_enqueue(notice["sdk_correlation"])
        if arrival is None or arrival[0] in self.steps or producer not in self.submissions:
            return None

        _, enqueued = arrival
        step = self.step_at(producer, taken)
        if step is None or step.modeled or not enqueued <= taken < completed <= step.end_ns:
            return None

        entry = self.progress(producer, step, taken)
        retired = self.progress(producer, step, completed)
        if entry is None or retired.fixed_prefix_ns - entry.fixed_prefix_ns != completed - taken:
            return None

        # Hold the external command's observed entry and advance with measured serial service to retirement.
        return HostWait(stopped, retired, resumed - completed, take_entry=entry)

    @staticmethod
    def protocol_partitions(events, calls, stamps, order, work):
        paths = defaultdict(list)
        for call in sorted(calls, key=lambda call: call.counters):
            first, last = call.counters
            index = bisect_right(events, first, key=lambda event: event[0])
            if not 0 < index < len(events):
                continue
            _, begin, source = events[index - 1]
            boundary, end, target = events[index]
            pair = source, target
            entered, returned = stamps[call.entered], stamps[call.returned]

            # Native protocol calls own their CPU service only in the unmodeled part of the actor path.
            if (last >= boundary or pair not in order or pair in work or begin is None or end is None
                    or not begin <= entered <= returned <= end):
                continue
            points = paths[pair]
            if points and entered < points[-1][0]:
                raise ValueError("native CPU protocol intervals overlap on one host thread")
            points.append((entered, call.entered[1]))
            if call.posted is not None:
                points.append((stamps[call.posted], call.posted[1]))
            points.append((returned, call.returned[1]))
        return paths

    def project(self, evidence, graph, host_fences, omitted, details, calls, stamps):
        queues = HostQueueEvidence(evidence, self.active)
        work = {(row["start"], row["finish"]): row["kind"] for row in graph["work"]}
        threads = {lane: int(resource["identity"].split("/")[1]) for lane, resource in enumerate(graph["resources"])
                   if resource["kind"] == "thread" and resource["serial"]}
        self.call_windows = self.synchronous_calls(graph, threads)
        lanes, positions = defaultdict(list), {}
        for event, (node, point) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True)):
            lane = node["resource"]
            if lane in threads:
                lanes[lane].append((point["order"], point["time_ns"], event))
                positions[threads[lane], point["order"]] = event

        report, arrivals, read_wakes = queues.observe()
        report["scheduler_trace"] = self.active.report
        notifications = defaultdict(list)
        for row in report["notifications"]:
            notifications[row["consumer"]].append(row)
        self.submissions = {
            thread for lane, thread in threads.items() if graph["resources"][lane]["role"] == "submission"
        }
        delays, counts = queues.handoffs(graph, positions, arrivals)
        waits, idle_inputs, idle = [], {}, {}
        self.steps = {thread: [] for thread in threads.values()}
        self.thread_bounds = {}
        protocol, by_thread = {}, defaultdict(list)
        for call in calls:
            if call.execution["in_flow"]:
                protocol.update(((start, finish), None) for start, finish, _ in call.services())
                by_thread[call.thread].append(call)
        order = (
            {(edge["source"], edge["target"]) for edge in graph["edges"] if edge["kind"] == "order"}
            if protocol
            else set()
        )
        for lane, events in lanes.items():
            events.sort()
            thread = threads[lane]
            self.thread_bounds[thread] = events[0][1], events[-1][1], events[-1][2]
            if thread == self.control_thread:
                input_waits = self.idle_waits(graph, events, work, notifications.get(thread, ()))
                idle = {(asleep, resumed): read for (_, _, asleep, resumed), read in input_waits.items()}
            paths = self.protocol_partitions(events, by_thread[thread], stamps, order, work)
            for (_, begin, source), (_, end, target) in pairwise(events):
                edge = source, target
                kind = work.get(edge)
                if begin is None or end is None or begin >= end or kind not in {None, "host"} or edge in host_fences:
                    continue
                points = [(begin, source), *paths.get(edge, ()), (end, target)]
                for (entered, start), (returned, finish) in pairwise(points):
                    endpoints = start, finish
                    modeled = kind == "host" or endpoints in protocol
                    service, conditions = self.host_time(thread, entered, returned, modeled=modeled,
                                                         submission=thread in self.submissions)
                    if conditions:
                        delays.setdefault(endpoints, {}).update(conditions)
                    self.steps[thread].append(HostStep(start, finish, entered, returned, modeled, service))
                    if endpoints in protocol:
                        protocol[endpoints] = service

        bound, kernel, external = 0, 0, 0
        for tid, host in self.steps.items():
            queue_waits = {tuple(row["off_cpu_ns"]): row for row in notifications.get(tid, ())}
            for step in host:
                begin, end = step.begin_ns, step.end_ns
                for asleep, resumed in self.active.within(self.active.sleep.get(tid, ()), begin, end):
                    if not begin <= asleep < resumed <= end:
                        continue
                    stopped = self.progress(tid, step, asleep)
                    if stopped is None:
                        continue
                    if tid == self.control_thread and (read := idle.get((asleep, resumed))):
                        read_end, delivered = read

                        # A delivered input binds an external release. An empty poll retains its measured wait duration.
                        if delivered:
                            idle_inputs[read_end] = HostWait(stopped, ExternalWake(resumed), 0)
                        else:
                            waits.append(HostWait(stopped, None, resumed - asleep))
                        continue
                    notification = queue_waits.get((asleep, resumed))
                    if notification is not None:
                        wait = self.queue_wait(notification, stopped, positions, queues)
                    else:
                        blocked = asleep, resumed
                        wait = self.kernel_wait(tid, blocked, stopped, read_wake=read_wakes.get((tid, blocked)))
                    if wait is None:
                        continue

                    waits.append(wait)
                    bound += notification is not None
                    kernel += notification is None
                    external += isinstance(wait.source, KernelWake) and wait.source.local_thread not in self.steps

        report["bound_to_fixed_work"] = bound
        report["kernel_wakes_bound"] = kernel
        report["external_kernel_wakes_bound"] = external
        report.update(counts)
        if not details:
            del report["notifications"]
        posts = {}
        for record in omitted:
            boundary = positions.get((record.thread, record.order))
            if boundary is None:
                boundary = self.progress_at(record.thread, record.posted_ns)
            if boundary is not None:
                posts[record] = boundary
        return delays, waits, idle_inputs, posts, report, protocol
