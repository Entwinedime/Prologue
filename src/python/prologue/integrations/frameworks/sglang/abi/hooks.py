"""Install process-wide native hooks at plugin load, or scope a replacement to one call."""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial

from prologue.integrations.frameworks.sglang.abi.registry import abi, resolve


@dataclass(frozen=True, slots=True)
class HookSpec:
    target: str
    callback: Callable
    mode: str = "around"

    def __post_init__(self):
        if self.mode not in {"around", "replace"} or not callable(self.callback):
            raise ValueError(f"invalid {self.mode} hook for {self.target}")
        if not all(part.isidentifier() for part in self.target.split(".")):
            raise ValueError(f"invalid native hook target {self.target!r}")


class HookTable:
    def __init__(self, *specs):
        self.specs = specs

    @classmethod
    def group(cls, callbacks):
        return cls(*(HookSpec(target, callback) for target, callback in callbacks.items()))

    def install(self):
        profile = abi()
        if "native_hooks" not in profile.capabilities:
            raise ValueError(f"native hooks are unavailable in revision {profile.revision}")
        registry = resolve(profile.symbols.hook_registry)
        modes = resolve(profile.symbols.hook_type)

        # Native plugin loading defers target imports until all plugins have registered.
        for spec in self.specs:
            target = profile.hook_paths.get(spec.target, spec.target)
            registry.register(target, spec.callback, modes(spec.mode))

    @classmethod
    def around(cls, target, callback):
        cls(HookSpec(target, callback)).install()

    @classmethod
    def replace(cls, target, callback):
        cls(HookSpec(target, callback, "replace")).install()

    @staticmethod
    def apply():
        resolve(abi().symbols.hook_registry).apply_hooks()

    @staticmethod
    @contextmanager
    def scoped(instance, *specs):
        # Restoring an inherited method means removing the instance override.
        with ExitStack() as restore:
            for spec in specs:
                name = spec.target
                original = getattr(instance, name)
                local = vars(instance).get(name)
                inherited = name not in vars(instance)
                callback = spec.callback if spec.mode == "replace" else partial(spec.callback, original)
                setattr(instance, name, callback)
                if inherited:
                    restore.callback(delattr, instance, name)
                else:
                    restore.callback(setattr, instance, name, local)
            yield
