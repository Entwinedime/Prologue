"""Identify physical pages at the native Radix cache's ownership boundaries."""


class RadixPages:
    @staticmethod
    def require(cache):
        from sglang.srt.mem_cache.radix_cache import RadixCache

        if type(cache) is not RadixCache or cache.disable or cache.is_eagle or cache.eviction_policy != "lru":
            raise ValueError("prefix caching requires the native dense LRU Radix cache")

    @staticmethod
    def indices(tokens, page_size):
        # Native physical indices begin after the dummy page.
        return [token // page_size - 1 for token in tokens[::page_size].tolist()]

    @staticmethod
    def install(**hooks):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        target = "sglang.srt.mem_cache.radix_cache.RadixCache."
        for method, function in hooks.items():
            HookRegistry.register(target + method, function, HookType.AROUND)
