"""Read a fixed, portable pair of template and service-cost publications."""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from prologue.common.cost import cost_identity
from prologue.common.io import content_id, read_json, write_json


def asset_path(directory, name):
    path = Path(name)
    resolved = (directory / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(directory):
        raise ValueError(f"asset reference escapes its library: {name}")
    return resolved


@dataclass(frozen=True, slots=True)
class BoundTemplates:
    """Keep the chosen template files and their action layouts together for one prediction."""

    paths: list[Path]
    definitions: Path | None
    headers: list[dict]
    worker_count: int
    worker_deployment: dict
    recordings: tuple[str, ...]


class TemplateCatalog:
    """Resolve worker bindings and compatible reusable templates against the released observation catalog."""

    def __init__(self, directory, index):
        self.directory = directory
        self.observations = index["observations"]
        self.body_digests = {}
        self.bindings = {}

        for binding in index["bindings"]:
            key = cost_identity(binding["domain"])
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
        body = read_json(asset_path(self.directory, shard["body"]))
        for position, definition in enumerate(body["definitions"]):
            if "object" in definition:
                key = definition["object"]
                path = asset_path(self.directory / "definitions", key + ".json")
                value = read_json(path)
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
                recipe = read_json(sampling) if sampling is not None else None
                shard = cls.write_shard(directory, name, read_json(path), sampling=recipe)
                shards.append(shard)

            records[identity] = {"deployment": deployment, "shards": shards}

        document = {"bindings": bindings, "observations": records}
        cls(directory, document)
        write_json(directory / "index.json", document, replace=False)

    def select(self, domain):
        binding = self.bindings.get(cost_identity(domain))
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
                    yield identity, read_json(asset_path(self.directory, shard["sampling"]))

    def program_identity(self, binding):
        for path in binding.paths:
            if path not in self.body_digests:
                self.body_digests[path] = sha256(path.read_bytes()).hexdigest()

        # The body includes the content addresses of its definitions. The header supplies the worker's initial state.
        return [
            binding.worker_count,
            binding.worker_deployment,
            binding.headers,
            [self.body_digests[path] for path in binding.paths],
        ]


class LibraryRelease:
    """Pin a single release for the lifetime of a reader so a concurrent publication cannot change its assets."""

    @classmethod
    def publish(cls, directory, observations, bindings, costs):
        from prologue.host.performance.fitting import FittedCosts

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
            current = read_json(pointer)["release"]
            self.directory = asset_path(self.root, current)
        manifest = self.directory / "release.json"

        if not manifest.exists() and self.directory == self.root and (not self.root.exists() or all(
            path.name in {".library.lock", ".pending.json", "releases"} for path in self.root.iterdir()
        )):
            self.identity = None
            self.catalog = None
            self.costs = None
            return

        published = read_json(manifest)
        self.identity = published["id"]
        self.catalog = asset_path(self.directory, published["templates"])
        self.costs = asset_path(self.directory, published["costs"])

    @cached_property
    def templates(self):
        if self.catalog is None:
            return TemplateCatalog(self.directory, {"bindings": [], "observations": {}})
        return TemplateCatalog(self.catalog.parent, read_json(self.catalog))

    @cached_property
    def cost_identities(self):
        if self.costs is None:
            return {}
        return {content_id(group["domain"]): content_id([group["definitions"], group["models"]])
                for group in read_json(self.costs)["domains"]}

    def summarize(self) -> dict:
        """Describe the templates a reader can select and the ranges of the published cost models."""
        templates = self.templates
        bindings = []
        definitions = set()

        for binding in templates.bindings.values():
            selected = templates.select(binding["domain"])
            active = set()
            for path in selected.paths:
                for definition in read_json(path)["definitions"]:
                    key = definition.get("object")
                    active.add(key if key is not None else content_id(definition))

            definitions.update(active)

            workers = []
            for header in selected.headers[:selected.worker_count]:
                rank = header["resources"][0]["rank"]
                roles = header["domain"]["worker_roles"]
                workers.append({"rank": rank, "roles": [role["parallel"] for role in roles.values()
                                                       if role["representative"] == rank]})

            bindings.append({"domain": binding["domain"], "workers": workers, "templates": len(active),
                             "recordings": list(selected.recordings)})

        costs = read_json(self.costs) if self.costs is not None else {"domains": [], "sample_recordings": []}
        coverage = []
        for domain in costs["domains"]:
            families = Counter()
            extents = {}
            for model in domain["models"]:
                families[model["family"]] += 1
                for axis, (low, high) in model["bounds"].items():
                    left, right = extents.get(axis, (low, high))
                    extents[axis] = [min(left, low), max(right, high)]

            # Extents describe the collection. Individual models retain their own possibly disjoint coverage.
            coverage.append({"domain": domain["domain"], "models": len(domain["models"]),
                             "families": dict(families), "feature_extents": extents})

        return {"release": self.identity, "template_count": len(definitions), "bindings": bindings,
                "cost_domains": coverage, "sample_tables": len(costs["sample_recordings"]),
                "cost_recordings": sorted({origin for origins in costs["sample_recordings"] for origin in origins})}
