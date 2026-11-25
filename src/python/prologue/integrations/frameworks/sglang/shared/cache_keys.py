"""Apply routed content to the native radix key at its construction boundary."""

from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import symbols


class RoutedCacheKeys:
    def install_cache_keys(self):
        HookTable.around(symbols().radix_key_init, self.cache_key)

    def cache_key(self, original, key, token_ids, extra_key=None, is_bigram=False, limit=None):
        if extra_key is not None:
            token_ids = self.routing.key(extra_key, token_ids if limit is None else token_ids[:limit])
            extra_key = None
        original(key, token_ids, extra_key, is_bigram, limit)
