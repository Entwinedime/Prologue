"""Interpret the logical workers and profiler windows published by a native capture."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProfileWindow:
    observed: tuple[int, ...]
    replay: bool = False

    def protocol_directory(self, capture: Path) -> Path:
        root = capture / "protocol"
        return root / f"worker-{self.observed[0]}" if self.replay else root


class CapturedWorkers:
    def __init__(self, instances):
        self.owners = {}
        self.observed = []
        self.windows = {}
        self.joint_execution = True

        for instance in instances:
            owner = instance["id"]
            participants = instance["native_participants"]
            observed = instance["observed_roles"]
            if not isinstance(owner, str) or not owner or owner in self.windows or not participants:
                raise ValueError("a capture needs unique instances with participating workers")
            for rank in participants:
                if type(rank) is not int or rank < 0 or rank in self.owners:
                    raise ValueError("native participants must be distinct nonnegative integer workers")
                self.owners[rank] = owner
            if (any(type(rank) is not int or rank not in participants for rank in observed)
                    or len(set(observed)) != len(observed)):
                raise ValueError("observed workers must be distinct participants of their own instance")
            self.observed.extend(observed)

            # Placement qualifies peer timing. Replay windows identify which protocol records accompany a profile.
            placement = instance.get("placement", "full")
            if placement not in {"full", "single_device"}:
                raise ValueError(f"unknown worker placement: {placement}")
            replays = instance.get("profile_windows")
            if replays is None:
                windows = [ProfileWindow(tuple(observed))] if observed else []
            else:
                if (any(len(row) != 1 or type(row[0]) is not int for row in replays)
                        or sorted(row[0] for row in replays) != sorted(observed)):
                    raise ValueError("worker replay windows must cover each profiled worker exactly once")
                windows = [ProfileWindow(tuple(row), replay=True) for row in replays]
            self.windows[owner] = windows
            if placement != "full" or replays is not None:
                self.joint_execution = False

        if not self.owners:
            raise ValueError("a native capture requires participating workers")
