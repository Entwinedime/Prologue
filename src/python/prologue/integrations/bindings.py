"""Bind one installed framework, its ABI revision and a device platform to a job."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

from prologue.common.interfaces import DeviceRuntime, PlatformBackend, PlatformDomain


@dataclass(frozen=True)
class Route:
    framework: str
    platform: str
    revision: str

    def __post_init__(self):
        if not all(name.isidentifier() and not name.startswith("_") for name in (self.framework, self.platform)):
            raise ValueError("adapter names must be package identifiers")
        if not self.revision:
            raise ValueError("a framework revision is required")

    def framework_module(self, component):
        return import_module(f"prologue.integrations.frameworks.{self.framework}.{component}")

    def platform_module(self):
        return import_module(f"prologue.integrations.platforms.{self.platform}.backend")


_binding = ContextVar("prologue_route", default=None)


@contextmanager
def use_route(selection):
    token = _binding.set(selection)
    try:
        yield selection
    finally:
        _binding.reset(token)


@lru_cache
def _image_route(framework, platform, revision):
    return Route(framework, platform, revision)


def route() -> Route:
    selection = _binding.get()
    if selection is not None:
        return selection

    # A fresh native child inherits its image declaration, including across spawn.
    return _image_route(*(os.environ[key] for key in
                          ("PROLOGUE_FRAMEWORK", "PROLOGUE_PLATFORM", "PROLOGUE_FRAMEWORK_REVISION")))


def device_runtime() -> DeviceRuntime:
    return route().platform_module().device


def platform_backend() -> PlatformBackend:
    return route().platform_module().platform


@lru_cache
def _domain_from_wire(wire):
    fields = json.loads(wire)
    fields["strided_attention"] = tuple(fields["strided_attention"])
    fields["native_symbols"] = tuple(tuple(row) for row in fields["native_symbols"])
    return PlatformDomain(**fields)


def platform_domain() -> PlatformDomain:
    # Control processes consume only these values and never load a device package.
    wire = os.environ.get("PROLOGUE_PLATFORM_DOMAIN")
    if wire:
        domain = _domain_from_wire(wire)
        if domain.name != route().platform:
            raise ValueError("the control platform differs from the job declaration")
        return domain
    return device_runtime().domain


class RuleName:
    def __init__(self, suffix, *, platform_only=False):
        self.suffix, self.platform_only = suffix, platform_only

    def __get__(self, instance, owner):
        selection = route()
        prefix = selection.platform if self.platform_only else f"{selection.framework}.{selection.platform}"
        return f"{prefix}.{self.suffix}"
