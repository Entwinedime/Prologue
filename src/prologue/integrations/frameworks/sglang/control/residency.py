"""Apply native cache decisions to the core's physical page pool."""

from functools import partial

from prologue.integrations.frameworks.sglang.radix import RadixPages


class PrefixResidency:
    def __init__(self, peer, cache):
        self.peer, self.cache = peer, cache
        RadixPages.require(cache)

    def store(self, original, cache, node, *args, **kwargs):
        result = original(cache, node, *args, **kwargs)
        if cache is self.cache:
            self.peer.call("retain_prefix", pages=RadixPages.indices(node.value, cache.page_size))
        return result

    def pin(self, acquire, original, cache, node, *args, **kwargs):
        result = original(cache, node, *args, **kwargs)
        if cache is self.cache and node is not cache.root_node:
            pages = []
            while node is not cache.root_node:
                pages.extend(RadixPages.indices(node.value, cache.page_size))
                node = node.parent
            delta = self.peer.call("pin_prefix", pages=pages, acquire=acquire)
            if delta != result.delta:
                raise ValueError("native cache locks disagree with physical page references")
        return result

    def rebind(self, original, cache, request, *args, **kwargs):
        result = original(cache, request, *args, **kwargs)
        if cache is self.cache and request.cache_protected_len:
            pages = RadixPages.indices(request.prefix_indices[:request.cache_protected_len], cache.page_size)
            self.peer.call("rebind_prefix", request=request.rid, pages=pages)
        return result

    def install(self):
        RadixPages.install(_record_store_event=self.store, inc_lock_ref=partial(self.pin, True),
                           dec_lock_ref=partial(self.pin, False), cache_unfinished_req=self.rebind)
