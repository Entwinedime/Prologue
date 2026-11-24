"""Publish immutable local assets and pin readers to one release."""

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Snapshot:
    manifest: dict
    documents: dict


class Store:
    families = ("templates", "performance", "rules")

    def __init__(self, root: Path):
        self.root = root

    def current(self):
        try:
            value = (self.root / "current").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return self.identity(value)

    @staticmethod
    def identity(value):
        if str(UUID(value)) != value:
            raise ValueError("release and asset references must be canonical UUIDs")
        return value

    def release(self, revision):
        if revision == "current":
            revision = self.current()
            if revision is None:
                raise ValueError("the store has no published release")
        path = self.root / "releases" / f"{self.identity(revision)}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["release"] != revision or manifest["assets"].keys() != set(self.families):
            raise ValueError("the published manifest does not match its release")
        return manifest

    def asset(self, release, family):
        identity = self.identity(release["assets"][family])
        path = self.root / family / f"{identity}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write(path, value):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def sync_directory(path):
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def snapshot(self, base):
        if base is None:
            return Snapshot({"release": None, "assets": {}}, {})
        manifest = self.release(base)
        return Snapshot(manifest, {family: self.asset(manifest, family) for family in self.families})

    def publish(self, previous, candidates, rule_review=None, operation=None):
        if candidates.keys() != set(self.families):
            raise ValueError("a release requires templates, performance and rules")
        base = previous.manifest["release"]
        assets = previous.manifest["assets"].copy()
        sources = {}
        for family, value in candidates.items():
            if family in assets and previous.documents[family] == value:
                continue
            assets[family] = str(uuid4())
            sources[family] = value

        revision = str(uuid4()) if sources else base
        receipt = {"release": revision, "base": base, "assets": assets}
        if rule_review is not None:
            receipt["rule_review"] = rule_review
        if operation is not None:
            receipt["operation"] = self.identity(operation)
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".candidate-", dir=self.root) as temporary:
            staging = Path(temporary)
            if sources:
                for family, value in sources.items():
                    self.write(staging / family, json.dumps(value, separators=(",", ":")))
                self.write(staging / "release", json.dumps(receipt, separators=(",", ":")))
                self.write(staging / "current", revision + "\n")

            with (self.root / ".publish.lock").open("a", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = self.current()
                if current != base:
                    if operation is not None and current is not None:
                        existing = self.release(current)
                        if (existing.get("operation") == operation and existing["base"] == base
                                and self.snapshot(current).documents == candidates):
                            return {"status": "published", **existing}
                    return {"status": "stale_release", "release": current, "base": base}
                if not sources:
                    return {"status": "unchanged", **previous.manifest}

                # Commit immutable objects and their manifest before replacing the only mutable pointer.
                for family in sources:
                    directory = self.root / family
                    directory.mkdir(exist_ok=True)
                    os.replace(staging / family, directory / f"{assets[family]}.json")
                    self.sync_directory(directory)
                releases = self.root / "releases"
                releases.mkdir(exist_ok=True)
                os.replace(staging / "release", releases / f"{revision}.json")
                self.sync_directory(releases)
                self.sync_directory(self.root)
                os.replace(staging / "current", self.root / "current")
                self.sync_directory(self.root)
        return {"status": "published", **receipt}
