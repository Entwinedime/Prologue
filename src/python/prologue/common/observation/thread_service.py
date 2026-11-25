"""Measure on-CPU service from native thread switches and declared excluded intervals."""

from bisect import bisect_right
from collections import defaultdict
from itertools import pairwise


def merged(ranges):
    result = []
    for begin, end in sorted(ranges):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((begin, end))
    return result


class ThreadService:
    def __init__(self, rows, dequeues, excluded, unavailable=()):
        self.activity = {}
        self.excluded = self.by_thread(excluded)
        self.unavailable = self.by_thread(unavailable)
        self.dequeues = self.by_thread(dequeues)
        self.loss = []
        for row in rows:
            thread = row["thread"]
            switches = row["switches"]
            if row["lost"] or any(left[0] > right[0] or left[1] == right[1] for left, right in pairwise(switches)):
                self.loss.append(thread)
                continue
            if not switches:
                continue
            begin, end = row["window_ns"]
            running = not switches[0][1]
            cursor = begin
            intervals = []
            for stamp, next_state in switches:
                if stamp > end:
                    break
                if stamp >= begin:
                    if running:
                        intervals.append((cursor, stamp))
                    cursor = stamp
                running = next_state
            if running:
                intervals.append((cursor, end))
            self.activity[thread] = (begin, end, intervals)

    @staticmethod
    def by_thread(intervals):
        ranges = defaultdict(list)
        for thread, begin, end in intervals:
            ranges[thread].append((begin, end))
        return {thread: merged(values) for thread, values in ranges.items()}

    @staticmethod
    def intersect(first, second):
        left, right = iter(first), iter(second)
        a, b = next(left, None), next(right, None)
        while a is not None and b is not None:
            begin, end = max(a[0], b[0]), min(a[1], b[1])
            if begin < end:
                yield begin, end
            if a[1] <= b[1]:
                a = next(left, None)
            else:
                b = next(right, None)

    @staticmethod
    def within(intervals, begin, end):
        first = bisect_right(intervals, begin, key=lambda interval: interval[1])
        last = bisect_right(intervals, end - 1, lo=first, key=lambda interval: interval[0])
        return intervals[first:last]

    def running_intervals(self, thread, begin, end):
        if thread not in self.activity or end <= begin:
            return None
        opened, closed, active = self.activity[thread]
        if not opened <= begin < end <= closed:
            return None
        return self.intersect(self.within(active, begin, end), [(begin, end)])

    def duration(self, thread, begin, end, submission):
        if self.within(self.unavailable.get(thread, ()), begin, end):
            return None
        covered = self.running_intervals(thread, begin, end)
        if covered is None:
            return None
        if submission:

            # An idle or spinning submission thread between two queue commands performs no modeled work.
            covered = self.intersect(covered, self.within(self.dequeues.get(thread, ()), begin, end))
        covered = list(covered)
        occupied = sum(y - x for x, y in covered)
        excluded = sum(
            y - x for x, y in self.intersect(covered, self.within(self.excluded.get(thread, ()), begin, end))
        )
        return occupied - excluded
