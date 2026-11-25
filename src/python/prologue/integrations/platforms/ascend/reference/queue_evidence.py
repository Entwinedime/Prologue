"""Prove a device task's release against recorded SDK slots and physical driver posts."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import cached_property
from itertools import pairwise

from prologue.integrations.platforms.ascend.trace_store import device_tasks, driver_submissions, physical_slots


class QueueEvidence:
    def __init__(self, runtime, apis, rows, flips):
        self.apis, self.tasks, self.flips = apis, device_tasks(rows), flips
        self.submissions = list(driver_submissions(runtime))
        by_caller, self.issued, self.post_index = defaultdict(list), defaultdict(list), {}
        for stream, post in self.submissions:
            by_caller[post.thread, stream].append(post)
            if post.record["tasks"]:
                self.post_index[id(post)] = len(self.issued[stream])
                self.issued[stream].append(post)
        self.by_caller = {key: ([p.begin for p in posts], posts) for key, posts in by_caller.items()}
        self.resolved = {}

    def inside(self, api, stream):
        starts, posts = self.by_caller.get((api.thread, stream), ((), []))
        contained = posts[bisect_left(starts, api.begin):bisect_right(starts, api.end)]
        if any(post.end > api.end for post in contained):
            raise ValueError("driver post escapes its recorded API interval")
        return contained

    @cached_property
    def slots(self):
        return physical_slots(self.tasks, self.flips)

    def posts(self, connection, stream):
        return [post for post in self.inside(self.apis[connection], stream) if post.record["tasks"]]

    def successive_posts(self, posts):
        return all(self.post_index[id(right)] == self.post_index[id(left)] + 1 and left.end <= right.begin
                   for left, right in pairwise(posts))

    @staticmethod
    def consecutive(slots):
        return all(
            (batch == following and successor == task + 1) or (following == batch + 1 and owners == {None})
            for ((batch, task), _), ((following, successor), owners) in pairwise(slots)
        )

    def neighbor(self, stream, position, post, side):
        issued = self.issued[stream]
        index = self.post_index[id(post)] + side
        slots, owners = self.slots[stream]
        if not 0 <= index < len(issued) or not 0 <= position < len(slots):
            return False
        tag = slots[position][1]
        if len(tag) != 1:
            return False
        connection, = tag
        if connection not in self.apis:
            return False
        group = self.posts(connection, stream)
        boundary = -1 if side < 0 else 0
        neighbor = issued[index]
        if not group or group[boundary] is not neighbor:
            return False
        if not self.successive_posts([neighbor, post] if side < 0 else [post, neighbor]):
            return False
        if len(group) == 1:
            return True
        tagged = owners[connection]
        block = slots[tagged[0]:tagged[-1] + 1]
        return (position == tagged[boundary] and len(block) == sum(p.record["tasks"] for p in group)
                and all(tags == tag for _, tags in block) and self.consecutive(block) and self.successive_posts(group))

    def prove_group(self, connection, stream, posts):
        # Each candidate interval consumes exactly the recorded SQE count, including proven batch flips.
        slots, by_owner = self.slots[stream]
        tagged = by_owner[connection]
        width = sum(post.record["tasks"] for post in posts)
        starts = []
        if posts and tagged and self.successive_posts(posts):
            for start in range(max(0, tagged[-1] - width + 1), min(tagged[0], len(slots) - width) + 1):
                end = start + width
                block = slots[start:end]
                if any(tags not in ({connection}, {None}) for _, tags in block) or not self.consecutive(block):
                    continue
                if width != len(tagged) and not (
                    self.neighbor(stream, start - 1, posts[0], -1)
                    and self.neighbor(stream, end, posts[-1], 1)
                    and self.consecutive(slots[start - 1:end + 1])
                ):
                    continue
                starts.append(start)
        if len(starts) != 1:
            raise ValueError(
                f"driver evidence cannot identify one release on stream {stream} for connection {connection}"
            )

        cursor, = starts
        releases = {}
        for post in posts:
            end = cursor + post.record["tasks"]
            for slot, tags in slots[cursor:end]:
                if connection in tags:
                    releases[slot] = post
            cursor = end
        return releases

    def release(self, task):
        stream, connection = int(task["Physic Stream Id"]), task["connection_id"]
        key = stream, connection
        if key not in self.resolved:
            posts = self.posts(connection, stream)
            if len(posts) == 1:
                return posts[0]
            self.resolved[key] = self.prove_group(connection, stream, posts)
        return self.resolved[key][int(task["Batch Id"]), int(task["Task Id"])]
