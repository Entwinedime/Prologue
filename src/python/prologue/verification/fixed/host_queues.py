"""Find actual software-queue notifications independently of the projected graph."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import chain, pairwise

from prologue.common.observation import ThreadService
from prologue.verification.core.adapters import platform_evidence


class HostQueueEvidence:
    def __init__(self, source, activity):
        self.source, self.activity = source, activity
        self.native = {}
        recorded = {"eventfd_read": [], "eventfd_write": []}

        for native in source.runtime:
            row = native.record
            if row["api"] in recorded:

                # Credits and wakes compare in RAW time. Queue attribution uses the existing counter interval.
                interval = platform_evidence().interval(
                    row["begin_raw_ns"], row["end_raw_ns"], row["fd"], row
                )
                self.native[id(interval)] = native
                recorded[row["api"]].append(interval)

        self.notifications = sum(map(len, recorded.values()))
        self.reads = recorded["eventfd_read"]
        self.read_index, self.write_index = (
            platform_evidence().thread_intervals(records) for records in recorded.values()
        )
        self.takes, self.completions = defaultdict(list), defaultdict(list)

        for correlation, command in source.dequeue.items():
            self.takes[command.thread].append(command)
            if source.window.begin <= command.begin <= command.end <= source.window.end:
                self.completions[command.thread].append((self.stamp(command, "end"), correlation))

        for commands in self.takes.values():
            commands.sort(key=lambda command: command.begin)
        for commands in self.completions.values():
            commands.sort()

    def producer(self, read, sleep, resume):
        if any(other is not read and other.begin < read.end and read.begin < other.end
               for other in self.read_index.overlapping(read)):
            return None

        # An eventfd read can consume several credits while its reader is off CPU.
        writes = [row for row in self.write_index.overlapping(read)
                  if row.begin < read.end and read.begin < row.end and row.record["value"] > 0]
        if not writes or sum(row.record["value"] for row in writes) != read.record["value"]:
            return None

        # The write that caused the wake can precede the switch-out record.
        wakes = self.activity.waking(read.record["thread"], read.begin, resume)
        if len(wakes) == 1:
            wake, = wakes
            matching = [write for write in writes if write.record["thread"] == wake.local_thread
                        and write.begin <= wake.raw_ns <= write.end]

            if len(matching) == 1:
                return matching[0], wake

        # With one write during the sleep, the owner is known even without an exact wake time.
        if not wakes and len(writes) == 1 and sleep <= writes[0].begin <= resume:
            return writes[0], None
        return None

    def stamp(self, command, side):
        return self.source.host_times[f"thread/{command.thread}", getattr(command, side)]

    def unowned_enqueue(self, correlation):
        command = self.source.enqueue[correlation]
        if self.source.owner(command) is None:
            return command.thread, self.stamp(command, "begin")
        return None

    def notified_take(self, read, write):
        takes = self.takes[read.thread]
        index = bisect_left(takes, read.end, key=lambda command: command.begin)
        if index == len(takes) or (index and takes[index - 1].end > read.begin):
            return None

        take = takes[index]
        producer = self.source.enqueue_ranges.enclosing(self.native[id(write)])
        if producer is None or producer.thread == read.thread or take.end > self.source.window.end:
            return None

        # The writer is inside the exact enqueue whose command the worker takes after the read.
        correlation = take.record["args"]["correlation_id"]
        if producer is self.source.enqueue[correlation]:
            return take, producer

    def observe(self):
        counts, notifications, arrivals, read_wakes = Counter(), [], set(), {}
        for read in self.reads:
            interval = self.native[id(read)]
            if not (self.source.window.begin <= interval.begin < interval.end <= self.source.window.end):
                continue
            if self.source.excluded_ranges.enclosing(interval) is not None:
                continue

            reader = interval.thread
            submission = reader in self.takes
            scope = self.source.owner(interval)
            if not submission:
                if scope is None or scope in self.source.closed_scopes:
                    continue
                counts["reads"] += 1

                if reader not in self.activity.sleep:
                    counts["unknown_thread_activity"] += 1
                    continue

            sleeps = self.blocked(read, reader)
            if not sleeps:
                continue
            if not submission:
                counts["blocked_reads"] += 1

            notice = self.producer(read, *sleeps[0]) if len(sleeps) == 1 else None
            if notice is None:
                if not submission:
                    counts["unpaired_notifications"] += 1
                continue

            write, wake = notice
            if wake is not None:
                key = reader, sleeps[0]
                self.source.require(read_wakes.setdefault(key, wake) == wake, "conflicting eventfd wake evidence")

            if submission:
                if pair := self.notified_take(interval, write):
                    take, post = pair
                    owner = self.source.owner(post)
                    if owner is not None and owner not in self.source.closed_scopes:
                        arrivals.add(take.record["args"]["correlation_id"])
                continue

            # The SDK closes its dequeue mark before the repository retires the command and notifies the waiter.
            producer = write.record["thread"]
            commands = self.completions.get(producer, ())
            index = bisect_right(commands, write.begin, key=lambda command: command[0]) - 1
            if index < 0 or producer == reader:
                counts["unpaired_queue_completions"] += 1
                continue

            finish, correlation = commands[index]
            take = self.source.dequeue[correlation]
            enqueue = self.source.enqueue[correlation]
            if self.source.excluded_ranges.enclosing(enqueue) is not None:
                counts["observer_producers"] += 1
                continue

            counts["dequeue_mark_before_wait"] += finish < read.begin
            released = wake.raw_ns if wake is not None else write.begin
            notifications.append({
                "scope": scope, "site": self.source.native["scopes"][scope]["site"],
                "descriptor": read.thread, "consumer": reader, "producer": producer,
                "sdk_correlation": correlation, "dequeue_order": take.end,
                "dequeue_started_ns": self.stamp(take, "begin"),
                "dequeue_finished_ns": finish,
                "wait_ns": [read.begin, read.end], "off_cpu_ns": list(sleeps[0]),
                "notification_ns": [released, min(write.end, sleeps[0][1])],
            })

        notifications.sort(key=lambda row: row["wait_ns"])
        return {
            "notification_records": self.notifications, **counts, "queue_completion_notifications": len(notifications),
            "queue_arrival_notifications": len(arrivals),
            "first": notifications[0] if notifications else None,
            "notifications": notifications,
        }, arrivals, read_wakes

    def blocked(self, read, thread):
        return [
            (first, last)
            for first, last in ThreadService.within(self.activity.sleep.get(thread, ()), read.begin, read.end)
            if read.begin <= first < last <= read.end
        ]

    def handoffs(self, graph, positions, arrivals):
        delays, counts = {}, Counter()
        edges = {(edge["source"], edge["target"]) for edge in graph["edges"] if edge["kind"] == "submission"}
        targets = {target for _, target in edges}
        events = graph["events"]
        ordered = ((edge["source"], edge["target"]) for edge in graph["edges"] if edge["kind"] == "order")
        work = ((interval["start"], interval["finish"]) for interval in graph["work"])
        serial_predecessors = {target for source, target in chain(ordered, work)
                               if target in targets and events[source]["resource"] == events[target]["resource"]}

        for takes in self.takes.values():
            for previous, take in pairwise([None, *takes]):
                correlation = take.record["args"]["correlation_id"]
                post = self.source.enqueue[correlation]
                sender = positions.get((post.thread, post.begin))
                target = positions.get((take.thread, take.begin))
                if post.thread == take.thread or (sender, target) not in edges:
                    continue

                if (previous is not None and post.begin < previous.end
                        and target in serial_predecessors and correlation not in arrivals):

                    # The retained serial predecessor supplies readiness for queued work.
                    continue

                elapsed = self.stamp(take, "begin") - self.stamp(post, "begin")
                self.source.require(elapsed >= 0, "observed queue take precedes its original enqueue")
                delays[sender, target] = {"queue_handoff_ns": elapsed}
                counts["notified_active_takes"] += correlation in arrivals

        return delays, counts
