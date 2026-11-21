"""Join HCCL notify endpoints to the device tasks that actually recorded or consumed them."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal


class HcclNotifications:
    def __init__(self, rows):
        self.tasks = {}
        for row in rows:
            if row.get("ph") != "X" or row.get("name") not in {"Notify_Record", "Notify_Wait"}:
                continue
            args = row["args"]
            key = args["stream id"], args["task id"], args["context id"], Decimal(str(row["ts"]))
            side = "record" if row["name"] == "Notify_Record" else "wait"
            endpoint = side, args["src rank"], args["dst rank"], args["notify_id"]

            # An SDK annotation and a device task must identify the same physical subtask, including timer epoch.
            self.tasks[key] = None if key in self.tasks else endpoint

    def observed(self, commands, clock):
        signals = []
        for command in commands:
            row = command.source.record
            args = row["args"]
            key = args["Physic Stream Id"], args["Task Id"], args["Subtask Id"], Decimal(str(row["ts"]))
            if key not in self.tasks:
                continue
            endpoint = self.tasks[key]
            if endpoint is None:
                return []

            # A record can signal after it starts. A wait can finish only after the signal has arrived.
            start, finish = clock.interval(row)
            signals.append([*endpoint, start if endpoint[0] == "record" else finish])
        return signals


@dataclass(frozen=True, slots=True)
class NotifyProgress:
    prefix: tuple[int, int]
    suffix: tuple[int, int]

    @classmethod
    def within(cls, rows, start, finish, clock):
        endpoints = defaultdict(list)
        for side, local, remote, handle, stamp in rows:
            if not start <= stamp <= finish:
                raise ValueError("HCCL notification falls outside its measured collective")
            endpoints[side, local, remote, handle].append(
                cls(clock.elapsed_bounds(start, stamp), clock.elapsed_bounds(stamp, finish))
            )
        return endpoints

    @staticmethod
    def peer_bounds(own, peer, lower, upper):
        for (side, local, remote, handle), steps in own.items():
            opposite = "record" if side == "wait" else "wait"
            matches = peer.get((opposite, remote, local, handle), ())
            if local == remote or len(steps) != len(matches):
                continue

            # One stream on each endpoint consumes each register in order within this matching native post.
            for step, match in zip(steps, matches):
                if side == "wait":
                    lower = max(lower, step.suffix[0] + match.prefix[0])
                else:
                    upper = min(upper, step.suffix[1] + match.prefix[1])
        return lower, upper
