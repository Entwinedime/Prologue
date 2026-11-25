"""Coordinate durable host task inputs, single writers and cumulative time reservations."""

import fcntl
import shutil
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from prologue.runtime.io import discard_path
from prologue.runtime.limits import AllowanceExhausted


@contextmanager
def file_lock(path: Path, *, blocking: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(handle, flags)
        yield


class BinarySnapshot:
    def __init__(self, source: Path):
        self.source = source.resolve(strict=True)
        self.digest = sha256(self.source.read_bytes()).hexdigest()

    def check(self, snapshot: Path):
        if sha256(snapshot.read_bytes()).hexdigest() != self.digest:
            raise ValueError(f"task executable differs from the requested binary: {snapshot}")

    def copy_to(self, snapshot: Path):
        shutil.copyfile(self.source, snapshot)
        self.check(snapshot)
        snapshot.chmod(0o500)


class TaskWorkspace:
    def __init__(self, root: Path):
        self.root = root

    @contextmanager
    def initialize(self, marker: str):
        """Under the task lock, expose a draft once and publish its completion marker last."""
        staging = self.root / ".staging"
        ready = staging / "ready"
        if ready.exists():
            self._publish(ready, marker)
        if (self.root / marker).exists():
            if staging.exists():
                staging.rmdir()
            yield None
            return
        if any(p.name not in {".lock", ".staging"} for p in self.root.iterdir()):
            raise ValueError(f"new task requires an empty directory: {self.root}")

        # A draft never reached publication. A ready directory can finish after any of its individual renames.
        draft = staging / "draft"
        discard_path(draft)
        draft.mkdir(parents=True)
        yield draft
        if not (draft / marker).exists():
            raise FileNotFoundError(draft / marker)
        draft.rename(ready)
        self._publish(ready, marker)
        staging.rmdir()

    def _publish(self, ready: Path, marker: str):
        for source in sorted(ready.iterdir(), key=lambda item: item.name == marker):
            target = self.root / source.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            source.rename(target)
        ready.rmdir()

    @staticmethod
    @contextmanager
    def reserve(state: dict, field: str, remaining_ns: int, save, *, elapsed_field: str | None = None, settled=None):
        if remaining_ns <= 0:
            raise AllowanceExhausted("cumulative time allowance exhausted")
        previous = state[field]
        state[field] = previous + remaining_ns
        save()

        # A killed coordinator leaves the entire reservation charged. An orderly exit records elapsed time.
        start = time.monotonic_ns()
        try:
            yield start + remaining_ns
        finally:

            # A failed teardown leaves the original reservation and recovery record intact on disk.
            if settled is None or settled():
                elapsed = time.monotonic_ns() - start
                state[field] = previous + elapsed
                if elapsed_field is not None:
                    state[elapsed_field] = elapsed
                save()
