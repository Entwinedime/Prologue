"""Resolve the framework ABI and device platform from explicit options and recorded domains."""

from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from pkgutil import iter_modules

from prologue.common.io import read_json
from prologue.host.libraries.release import LibraryRelease
from prologue.integrations import frameworks, platforms
from prologue.integrations.bindings import Route, use_route


@lru_cache
def installed(namespace):
    return tuple(name for _, name, package in iter_modules(namespace.__path__) if package and not name.startswith("_"))


def choose(label, explicit, observed, available):
    candidates = {explicit} if explicit else set(available)
    if observed:
        candidates.intersection_update(observed)
    candidates.intersection_update(available)
    if len(candidates) != 1:
        raise ValueError(f"cannot select one {label} from {sorted(candidates)}; specify --{label}")
    return candidates.pop()


class AdapterRegistry:
    @staticmethod
    @lru_cache
    def adapter(framework):
        if framework not in installed(frameworks):
            raise ValueError(f"unsupported framework {framework!r}")
        module = import_module(f"{frameworks.__name__}.{framework}.acquisition.bindings")
        return module.HostAdapter

    @classmethod
    def select(cls, framework, platform, version_profile):
        adapter = cls.adapter(framework)
        if platform not in installed(platforms):
            raise ValueError(f"unsupported platform {platform!r}")
        return Route(framework, platform, adapter.revision(version_profile))

    @classmethod
    @contextmanager
    def bind(cls, *, source=None, release=None, domains=(), framework=None, platform=None, version=None):
        source = source or {}
        records = list(domains)
        if release and release.identity:
            records.extend(shard["header"]["domain"] for row in release.templates.observations.values()
                           for shard in row["shards"])
        if source.get("framework") and framework and source["framework"] != framework:
            raise ValueError("the explicit framework disagrees with the input")
        framework = choose("framework", framework or source.get("framework"),
                           {r["framework"] for r in records if "framework" in r}, installed(frameworks))
        records = [r for r in records if r.get("framework", framework) == framework]
        declarations = {name: import_module(f"{platforms.__name__}.{name}.backend").device.domain
                        for name in installed(platforms)}

        def observed_platform(record):
            matches = {name for name, spec in declarations.items() if spec.runtime_domain_key in record}
            if label := record.get("platform"):
                if matches and label not in matches:
                    raise ValueError("recorded runtime key conflicts with its platform declaration")
                return {label}
            return matches

        if source.get("platform") and platform and source["platform"] != platform:
            raise ValueError("the explicit platform disagrees with the input")
        platform = choose("platform", platform or source.get("platform"),
                          {p for r in records for p in observed_platform(r)}, declarations)
        adapter = cls.adapter(framework)
        versions = {adapter.revision(v) for r in records if observed_platform(r) in (set(), {platform})
                    if (v := r.get("framework_revision") or r.get("implementation", {}).get("framework_revision"))}
        source_version = source.get("framework_revision")
        explicit = adapter.revision(version) if version else None
        if source_version:
            declared = adapter.revision(source_version)
            if explicit and explicit != declared:
                raise ValueError("the explicit framework revision disagrees with the input")
            explicit = declared
        if versions:
            explicit = choose("framework-version", explicit, versions, versions)
        selection = cls.select(framework, platform, explicit)
        with use_route(selection):
            yield selection, adapter

    @classmethod
    def command(cls, args):
        args.source = read_json(args.input) if getattr(args, "input", None) else None
        args.release = LibraryRelease(args.libraries) if getattr(args, "libraries", None) else None
        domains = []
        for path in getattr(args, "samples", ()):
            sample = read_json(path)
            if "domain" in sample:
                domains.append(sample["domain"])
        captures = getattr(args, "capture", None)
        for directory in captures if isinstance(captures, list) else [captures] if captures else []:
            manifest = directory / "execution.json"
            if manifest.is_file():
                domains.append(read_json(manifest)["inputs"])
        return cls.bind(source=args.source, release=args.release, domains=domains,
                        framework=getattr(args, "framework", None),
                        platform=getattr(args, "platform", None), version=getattr(args, "framework_version", None))
