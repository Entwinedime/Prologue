"""Discover contributions shipped in a trusted, installed package."""

from importlib import import_module
from pkgutil import iter_modules


def contributions(package, attribute):
    """Load declarations from the installed package in deterministic module order. The caller chooses the trusted
    namespace; input documents never supply import paths.
    """

    for module in sorted(iter_modules(package.__path__), key=lambda entry: entry.name):
        if not module.name.startswith("_"):
            component = import_module(f"{package.__name__}.{module.name}")
            yield from getattr(component, attribute)
