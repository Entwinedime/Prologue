"""Observe a named native attention call after SGLang has admitted its request and allocated KV."""

from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout


class DecodeAttentionObservation:
    def __init__(self, window, description):
        self.window = window
        self.directory = Path(description["output"])
        self.offset = description["rank_offset"]
        self.sites = description["attention_sites"]
        self.calls = defaultdict(int)
        self.rows = []

    def load_model(self, original, runner):
        result = original(runner)
        self.attach(runner)
        return result

    def attach(self, runner):
        self.rank = self.offset + runner.pp_rank * runner.tp_size + runner.tp_rank
        self.backend = runner.server_args.attention_backend
        modules = dict(runner.model.named_modules())
        selected = ModelScopes.selected(modules)
        sites = self.sites.get(str(self.rank), ())
        for site in sites:
            if selected.get(site) != "attention":
                raise ValueError(f"requested site is not an observed native attention module: {site}")
            module = modules[site]
            module.forward = partial(self.invoke, site, module, module.forward)

    def invoke(self, site, module, original, q, k, v, forward_batch, *args, **kwargs):
        import torch
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if (self.window is not None and self.window.phase != "source") or not forward_batch.forward_mode.is_decode():
            return original(q, k, v, forward_batch, *args, **kwargs)
        lengths = forward_batch.seq_lens_cpu
        context = tuple(int(length) - 1 for length in lengths.tolist()) if lengths is not None else ()
        query_rows = int(q.shape[0])
        key = site, tuple(sorted(context)), query_rows
        self.calls[key] += 1

        # Repeating the geometry measures the unprofiled envelope before recording device tasks.
        pool = get_attn_backend().token_to_kv_pool
        row = {
            "site": site,
            "implementation": f"{type(module).__module__}.{type(module).__qualname__}",
            "context": list(context),
            "query_rows": query_rows,
            "kv_layout": MhaKvLayout.from_pool(pool).document(),
            "backend": self.backend,
            "page_size": pool.page_size,
            "operands": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in (("argument.q", q), ("argument.k", k), ("argument.v", v))
                if tensor is not None
            },
            "full_context": not module.is_cross_attention and module.sliding_window_size < 0,
            "heads": {
                "query": module.tp_q_head_num,
                "key": module.tp_k_head_num,
                "value": module.tp_v_head_num,
                "key_dim": module.qk_head_dim,
                "value_dim": module.v_head_dim,
            },
        }
        profile = None
        if self.calls[key] == 2:
            from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks

            root = Path("cuts") / f"rank-{self.rank}" / str(len(self.rows))
            row["profile"] = str(root)
            profile = ScopedTasks.profile(self.directory / root, self.rank)
        begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        torch.npu.current_stream().synchronize()
        marker = torch.autograd.profiler.record_function if profile is not None else nullcontext
        with profile if profile is not None else nullcontext():
            with marker("prologue.sample.begin"):
                begin.record()
            with marker("prologue.sample.call"):
                result = original(q, k, v, forward_batch, *args, **kwargs)
            with marker("prologue.sample.end"):
                end.record()
            end.synchronize()
        row["envelope_ns"] = round(begin.elapsed_time(end) * 1_000_000)
        self.rows.append(row)
        return result

    def document(self):
        from prologue.integrations.platforms.ascend.markers import AscendMarkers

        return {"runtime": AscendMarkers.environment(), "calls": self.rows}

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        target = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
        HookRegistry.register(target, self.load_model, HookType.AROUND)
