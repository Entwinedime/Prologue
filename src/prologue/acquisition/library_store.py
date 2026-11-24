"""Publish both libraries together while readers keep their selected release."""

import fcntl
import json
import os
from pathlib import Path
from shutil import copytree, ignore_patterns, rmtree
from uuid import uuid4

from prologue.artifacts import write_json
from prologue.libraries import LibraryRelease, asset_path
from prologue.performance.fitting import FittedCosts


class LibraryStore:
    def __init__(self, root):
        self.root = root.resolve()

    def recover(self, release):
        journal = self.root / ".pending.json"
        if journal.is_file():
            pending = json.loads(journal.read_bytes())
            relative = Path(pending["release"])
            if relative.parent != Path("releases") or len(relative.name) != 32 or not all(c in "0123456789abcdef" for c in relative.name):
                raise ValueError("invalid pending library release")
            candidate = asset_path(self.root, relative)
            if release.directory != candidate and release.identity != pending["base"]:
                raise ValueError("pending update does not descend from the selected release")
            if candidate.exists() and candidate != release.directory:
                rmtree(candidate)
            journal.unlink()

    def open(self):
        if (self.root / ".pending.json").is_file():
            with (self.root / ".library.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.recover(LibraryRelease(self.root))
        return LibraryRelease(self.root)

    def publish(self, increment, before_commit=None, *, expected=None):
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "release.json").exists():
            raise ValueError("a fixed release is immutable; publish increments to a library store")
        with (self.root / ".library.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = LibraryRelease(self.root)
            self.recover(before)

            # A concurrent release may already cover the demand. The predictor rechecks it before another purchase or merge.
            if expected is not None and before.identity != expected.identity:
                return before
            receipts = json.loads((before.directory / "evidence.json").read_bytes()) if before.identity else {}
            if increment.applied(receipts):
                return before

            # Only the writer follows the journal. A reader pins the pair selected by current.json once.
            identity = uuid4().hex
            pointer = {"release": f"releases/{identity}"}
            journal = self.root / ".pending.json"
            write_json(journal, pointer | {"base": before.identity}, replace=False)
            candidate = self.root / pointer["release"]
            try:
                if before.identity:
                    copytree(before.directory, candidate, copy_function=os.link, ignore=ignore_patterns("release.json"))
                else:
                    write_json(candidate / "templates/index.json", {"bindings": [], "observations": {}})
                    FittedCosts.publish(candidate / "performance", None)

                # All changes replace files atomically. Unmodified bodies and samples share their old inodes.
                increment.apply(candidate, receipts)
                write_json(candidate / "evidence.json", receipts)
                write_json(candidate / "release.json", {
                    "id": identity, "templates": "templates/index.json", "costs": "performance/costs.json",
                })
                if before_commit:
                    before_commit(candidate)
                write_json(self.root / "current.json", pointer)
            finally:
                self.recover(LibraryRelease(self.root))
            return LibraryRelease(self.root)
