"""Read a fixed, portable pair of template and service-cost publications."""

import json
from hashlib import sha256
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from prologue.runtime.io import content_id, write_json


def asset_path(directory, name):
    path = Path(name)
    resolved = (directory / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(directory):
        raise ValueError(f"asset reference escapes its library: {name}")
    return resolved


@dataclass(frozen=True, slots=True)
class BoundTemplates:
    paths: list[Path]
    definitions: Path | None
    headers: list[dict]
    worker_count: int
    worker_deployment: dict
    recordings: tuple[str, ...]


class TemplateCatalog:
    def __init__(self, directory, index):
        self.directory = directory
        self.observations = index["observations"]
        self.body_digests = {}
        self.bindings = {}
        for binding in index["bindings"]:
            key = json.dumps(binding["domain"], sort_keys=True)
            if key in self.bindings:
                raise ValueError(f"template publication has conflicting worker bindings for {binding['domain']}")
            if {binding["workers"], *binding["imports"], *binding.get("sources", ())} - self.observations.keys():
                raise ValueError("worker binding refers to an unpublished observation")
            self.bindings[key] = binding

    @staticmethod
    def write_shard(directory, name, template, *, sampling=None):
        header = {key: template[key] for key in ("domain", "local_domain", "resources")}
        body = {key: value for key, value in template.items() if key not in header and key != "definitions"}
        references = []
        for position, definition in enumerate(template["definitions"]):
            key = content_id(definition)
            path = directory / "definitions" / (key + ".json")
            if not path.exists():
                write_json(path, definition, replace=False)
            references.append({"id": position, "object": key})

        # Frames keep their observed positions; equivalent roles point to one immutable definition.
        write_json(directory / name, body | {"definitions": references})
        shard = {"body": name, "header": header}
        if sampling is not None:
            shard["sampling"] = str(Path(name).with_suffix("")) + "-sampling.json"
            write_json(directory / shard["sampling"], sampling, replace=False)
        return shard

    def read_shard(self, shard):
        body = json.loads(asset_path(self.directory, shard["body"]).read_bytes())
        for position, definition in enumerate(body["definitions"]):
            if "object" in definition:
                key = definition["object"]
                path = asset_path(self.directory / "definitions", key + ".json")
                value = json.loads(path.read_bytes())
                if content_id(value) != key:
                    raise ValueError(f"published template definition differs from its content address: {key}")
                body["definitions"][position] = value | {"id": definition["id"]}
        return body

    @classmethod
    def publish(cls, directory, observations, bindings):
        records = {}
        for ordinal, (identity, (deployment, paths)) in enumerate(observations.items()):
            shards = []
            for role, (path, sampling) in enumerate(paths):
                name = f"bodies/{ordinal}/{role}.json"
                recipe = json.loads(sampling.read_bytes()) if sampling is not None else None
                shard = cls.write_shard(directory, name, json.loads(path.read_bytes()), sampling=recipe)
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
        recordings = tuple(dict.fromkeys((binding["workers"], *binding.get("sources", ()), *binding["imports"])))

        # Only an initialized worker supplies participants. Local observations can extend its effective program.
        primary = binding.get("compiled", workers["shards"])
        shards = [*primary, *(shard for name in binding["imports"] for shard in self.observations[name]["shards"])]
        definitions = self.directory / "definitions"
        return BoundTemplates(
            [asset_path(self.directory, shard["body"]) for shard in shards],
            definitions if definitions.is_dir() else None, [shard["header"] for shard in shards],
            len(workers["shards"]), workers["deployment"], recordings,
        )

    def sampling_sources(self):
        for identity, observation in self.observations.items():
            for shard in observation["shards"]:
                if "sampling" in shard:
                    yield identity, json.loads(asset_path(self.directory, shard["sampling"]).read_bytes())

    def program_identity(self, binding):
        for path in binding.paths:
            if path not in self.body_digests:
                self.body_digests[path] = sha256(path.read_bytes()).hexdigest()

        # The body includes the content addresses of its definitions. The header supplies the worker's initial state.
        return [binding.worker_count, binding.worker_deployment, binding.headers, [self.body_digests[path] for path in binding.paths]]


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
        self.root = directory.resolve()
        self.directory = self.root
        pointer = self.root / "current.json"
        if pointer.is_file():
            current = json.loads(pointer.read_bytes())["release"]
            self.directory = asset_path(self.root, current)
        manifest = self.directory / "release.json"
        if not manifest.exists() and self.directory == self.root and (not self.root.exists() or all(
            path.name in {".library.lock", ".pending.json", "releases"} for path in self.root.iterdir()
        )):
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

    @cached_property
    def cost_identities(self):
        if self.costs is None:
            return {}
        return {content_id(group["domain"]): content_id([group["definitions"], group["models"]])
                for group in json.loads(self.costs.read_bytes())["domains"]}
