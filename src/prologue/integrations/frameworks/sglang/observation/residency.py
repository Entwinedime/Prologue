"""Read the native cache's actual page storage and eviction effects."""

from functools import partial

from prologue.integrations.frameworks.sglang.radix import RadixPages


class CacheObservation:
    def __init__(self, observation):
        self.observation = observation
        self.tree = None
        self.reset()

    def reset(self):
        self.effects = []
        self.actions = 0

    def bind(self, tree):
        if self.tree is None:
            RadixPages.require(tree)
            self.tree = tree

    def record(self, operation, original, cache, node, *args, **kwargs):
        result = original(cache, node, *args, **kwargs)
        observation = self.observation
        if cache is self.tree and observation.capture_calls and observation.ingress:
            with observation.calls.scope("capture_auxiliary", "RadixCache.page_snapshot", operation, {}):
                pages = RadixPages.indices(node.value, cache.page_size)
            self.effects.append({"after_action": self.actions, "operation": operation, "pages": sorted(pages)})
        return result

    def install(self):
        RadixPages.install(_record_store_event=partial(self.record, "store"),
                           _record_remove_event=partial(self.record, "evict"))
