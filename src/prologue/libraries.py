"""Read a fixed, portable pair of template and service-cost publications."""

import json
from shutil import copyfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from prologue.artifacts import write_json


def asset_path(directory, name):
    path = Path(name)
    resolved = (directory / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(directory):
        raise ValueError(f"asset reference escapes its library: {name}")
    return resolved


@dataclass(frozen=True, slots=True)
class BoundTemplates:
    paths: list[Path]
    headers: list[dict]
    worker_count: int
    worker_deployment: dict
    recordings: tuple[str, ...]


class TemplateCatalog:
    def __init__(self, directory, index):
        self.directory = directory
        self.observations = index["observations"]
        self.bindings = {}
        for binding in index["bindings"]:
            key = json.dumps(binding["domain"], sort_keys=True)
            if key in self.bindings:
                raise ValueError(f"template publication has conflicting worker bindings for {binding['domain']}")
            if {binding["workers"], *binding["imports"]} - self.observations.keys():
                raise ValueError("worker binding refers to an unpublished observation")
            self.bindings[key] = binding

    @classmethod
    def publish(cls, directory, observations, bindings):
        records = {}
        for ordinal, (identity, (deployment, paths)) in enumerate(observations.items()):
            shards = []
            for role, (path, sampling) in enumerate(paths):
                body = json.loads(path.read_bytes())
                header = {field: body.pop(field) for field in ("domain", "local_domain", "resources")}

                # A frame names a definition independently of the other definitions present in this release.
                for identity_in_body, definition in enumerate(body["definitions"]):
                    definition["id"] = identity_in_body
                name = f"bodies/{ordinal}/{role}.json"
                write_json(directory / name, body, replace=False)
                shard = {"body": name, "header": header}
                if sampling is not None:
                    shard["sampling"] = f"bodies/{ordinal}/{role}-sampling.json"
                    copyfile(sampling, directory / shard["sampling"])
                shards.append(shard)
            records[identity] = {"deployment": deployment, "shards": shards}

        document = {"bindings": bindings, "observations": records}
        cls(directory, document)
        write_json(directory / "index.json", document, replace=False)

    def select(self, domain):
        binding = self.bindings.get(json.dumps(domain, sort_keys=True))
        if binding is None:
            return None
        workers = self.observations[binding["workers"]]
        recordings = tuple(dict.fromkeys((binding["workers"], *binding["imports"])))

        # Auxiliary bodies can extend local calls. Only the first observation supplies serving participants.
        shards = [shard for name in recordings for shard in self.observations[name]["shards"]]
        return BoundTemplates(
            [asset_path(self.directory, shard["body"]) for shard in shards],
            [shard["header"] for shard in shards], len(workers["shards"]), workers["deployment"], recordings,
        )

    def sampling_sources(self):
        for identity, observation in self.observations.items():
            for shard in observation["shards"]:
                if "sampling" in shard:
                    yield identity, json.loads(asset_path(self.directory, shard["sampling"]).read_bytes())


class LibraryRelease:
    @classmethod
    def publish(cls, directory, observations, bindings, costs):
        from prologue.performance.fitting import FittedCosts

        directory = directory.resolve()
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)

        # Make both libraries visible together. A failed import leaves no partially readable release.
        with TemporaryDirectory(dir=directory.parent, prefix=".publish-") as temporary:
            staged = Path(temporary) / "release"
            TemplateCatalog.publish(staged / "templates", observations, bindings)
            FittedCosts.publish(staged / "performance", costs)
            write_json(staged / "release.json", {
                "id": uuid4().hex, "templates": "templates/index.json", "costs": "performance/costs.json",
            })
            staged.rename(directory)
        return cls(directory)

    def __init__(self, directory):
        self.directory = directory.resolve()
        manifest = self.directory / "release.json"
        if not manifest.exists() and (not self.directory.exists() or not any(self.directory.iterdir())):
            self.identity = None
            self.catalog = None
            self.costs = None
            return

        published = json.loads(manifest.read_bytes())
        self.identity = published["id"]
        self.catalog = asset_path(self.directory, published["templates"])
        self.costs = asset_path(self.directory, published["costs"])

    @cached_property
    def templates(self):
        if self.catalog is None:
            return TemplateCatalog(self.directory, {"bindings": [], "observations": {}})
        return TemplateCatalog(self.catalog.parent, json.loads(self.catalog.read_bytes()))
