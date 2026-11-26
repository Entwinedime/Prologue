"""Select an installed ABI by native revision, declared alias or observed capabilities."""

from functools import lru_cache
from importlib import import_module
from pkgutil import iter_modules, resolve_name

from prologue.integrations.bindings import platform_domain, route
from prologue.integrations.frameworks.sglang.abi import profiles
from prologue.integrations.frameworks.sglang.abi.contract import NativeSymbols, SGLangABI
from prologue.integrations.frameworks.sglang.abi.profiles.rev_4e155ecc import profile as baseline

default_revision = baseline.revision


@lru_cache
def installed():
    return tuple(import_module(f"{profiles.__name__}.{name}").profile
                 for _, name, _ in iter_modules(profiles.__path__) if name.startswith("rev_"))


@lru_cache
def select(revision=None, capabilities=frozenset()) -> SGLangABI:
    if not revision and not capabilities:
        revision = default_revision
    candidates = [profile for profile in installed() if capabilities <= profile.capabilities and
                  (not revision or revision in profile.aliases or revision == profile.revision or
                   (len(revision) >= 8 and profile.revision.startswith(revision)))]
    if len(candidates) != 1:
        raise ValueError(f"cannot select one native ABI for {revision!r} and {sorted(capabilities)}")
    return candidates[0]


@lru_cache
def _for_platform(selection, domain):
    return select(selection.revision).for_platform(selection.framework, domain)


def abi() -> SGLangABI:
    return _for_platform(route(), platform_domain())


def symbols() -> NativeSymbols:
    return abi().symbols


def resolve(path):
    return resolve_name(path)
