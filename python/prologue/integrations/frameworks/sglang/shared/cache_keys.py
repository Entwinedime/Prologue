"""Apply routed content to the native radix key at its construction boundary."""


class RoutedCacheKeys:
    def install_cache_keys(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register("sglang.srt.mem_cache.radix_cache.RadixKey.__init__", self.cache_key, HookType.AROUND)

    def cache_key(self, original, key, token_ids, extra_key=None, is_bigram=False, limit=None):
        if extra_key is not None:
            token_ids = self.routing.key(extra_key, token_ids if limit is None else token_ids[:limit])
            extra_key = None
        original(key, token_ids, extra_key, is_bigram, limit)
