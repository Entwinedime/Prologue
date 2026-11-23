"""Record the same native attention boundary in serving and local sampling."""

from prologue.integrations.frameworks.sglang.batch import BatchWork


def implementation(module):
    return f"{type(module).__module__}.{type(module).__qualname__}"


class AttentionCall:
    def __init__(self, calls, module, site, *, require_parent):
        self.calls = calls
        self.module = module
        self.site = site
        self.require_parent = require_parent
        self.forward = module.forward

    def __call__(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        if self.require_parent and not self.calls.stack:
            return self.forward(q, k, v, forward_batch, save_kv_cache, **kwargs)

        parameters = {
            **BatchWork.attention(forward_batch),
            "query_heads": self.module.tp_q_head_num,
            "key_heads": self.module.tp_k_head_num,
            "value_heads": self.module.tp_v_head_num,
            "writes_kv": save_kv_cache,
        }
        with self.calls.scope("attention", implementation(self.module), self.site, parameters) as call:
            call["inputs"] = self.calls.ports(query=q, key=k, value=v)
            result = self.forward(q, k, v, forward_batch, save_kv_cache, **kwargs)
            call["outputs"] = self.calls.ports(output=result)
            return result
