"""Translate native serving calls into the shared observation contract."""

import json
import os
from functools import partial
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.acquisition.observation.calls import Calls


class Observation:
    def __init__(self, output, requests, detailed):
        self.output = output
        self.requests = requests
        self.publications = {native: [] for native in requests}
        self.finished = set()
        self.detailed = detailed
        self.active_batch = None
        if detailed:
            from torch.autograd.profiler import record_function

            self.calls = Calls(record_function)

    def flush(self, original, scheduler, empty_cache=True):
        # Drain logical KV state while preserving warmed device allocator workspaces.
        drained = original(scheduler, empty_cache=False)
        if drained and self.finished == self.publications.keys():
            self.save()
        return drained

    def publish(self, original, sender, output, *args, **kwargs):
        from sglang.srt.managers.io_struct import BatchTokenIDOutput

        result = original(sender, output, *args, **kwargs)
        if sender.socket is None or not isinstance(output, BatchTokenIDOutput):
            return result

        now = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        for rid, count, finished in zip(output.rids, output.completion_tokens, output.finished_reasons, strict=True):
            if rid not in self.publications or rid in self.finished:
                continue
            rows = self.publications[rid]
            if count > (rows[-1]["count"] if rows else 0):
                rows.append({"count": count, "raw_ns": now})
            if finished is not None:
                self.finished.add(rid)
        return result

    def save(self):
        document = {"publications": {self.requests[rid]: rows for rid, rows in self.publications.items()}}
        if self.detailed:
            document["calls"] = self.calls.rows

        # Only the output owner writes this file, after the observed publications.
        path = self.output / "worker.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
        pending.replace(path)

    def batch(self, original, scheduler, batch, *args, **kwargs):
        requests = [self.requests.get(request.rid) for request in batch.reqs]
        if not any(request is not None for request in requests):
            return original(scheduler, batch, *args, **kwargs)
        if any(request is None for request in requests):
            raise ValueError("warmup and observed requests occupied the same batch")

        if batch.forward_mode.is_decode():
            phase = "decode"
        elif batch.forward_mode.is_extend():
            phase = "prefill"
        else:
            raise ValueError(f"unsupported observed forward mode: {batch.forward_mode}")
        parameters = {"phase": phase, "members": len(requests)}
        with self.calls.scope("batch", "Scheduler.run_batch", "scheduler", parameters) as call:
            self.active_batch = call
            call["members"] = [
                {"request": identity, "slot": int(request.req_pool_idx), "kv_committed_tokens": request.kv_committed_len,
                 "kv_reserved_tokens": request.kv_allocated_len, "output_tokens": len(request.output_ids)}
                for identity, request in zip(requests, batch.reqs, strict=True)
            ]
            call["resources"] = self.resources(scheduler)
            return original(scheduler, batch, *args, **kwargs)

    @staticmethod
    def resources(scheduler):
        return {
            "slots": scheduler.req_to_token_pool.available_size(),
            "kv_tokens": scheduler.token_to_kv_pool_allocator.available_size(),
        }

    def process_result(self, original, scheduler, batch, result):
        if self.active_batch is None:
            return original(scheduler, batch, result)

        call = self.active_batch
        self.active_batch = None
        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        value = original(scheduler, batch, result)
        call["result_processing"] = {
            "begin_ns": begin, "end_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW),
            "resources": self.resources(scheduler),
        }
        return value

    @staticmethod
    def implementation(module):
        return f"{type(module).__module__}.{type(module).__qualname__}"

    def layer(self, original, module, site, positions, hidden_states, forward_batch, residual, *args, **kwargs):
        if not self.calls.stack:
            return original(positions, hidden_states, forward_batch, residual, *args, **kwargs)

        with self.calls.scope("layer", self.implementation(module), site, {}) as call:
            call["inputs"] = self.calls.ports(positions=positions, hidden=hidden_states, residual=residual)
            result = original(positions, hidden_states, forward_batch, residual, *args, **kwargs)
            call["outputs"] = self.calls.ports(hidden=result[0], residual=result[1])
            return result

    def attention(self, original, module, site, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        if not self.calls.stack:
            return original(q, k, v, forward_batch, save_kv_cache, **kwargs)

        parameters = {
            "phase": "decode" if forward_batch.forward_mode.is_decode() else "prefill",
            "seq_lens": forward_batch.seq_lens_cpu.tolist(),
            "query_heads": module.tp_q_head_num,
            "key_heads": module.tp_k_head_num,
            "value_heads": module.tp_v_head_num,
            "writes_kv": save_kv_cache,
        }
        with self.calls.scope("attention", self.implementation(module), site, parameters) as call:
            call["inputs"] = self.calls.ports(query=q, key=k, value=v)
            result = original(q, k, v, forward_batch, save_kv_cache, **kwargs)
            call["outputs"] = self.calls.ports(output=result)
            return result

    def projection(self, original, module, site, input_, *args, **kwargs):
        if not self.calls.stack:
            return original(input_, *args, **kwargs)

        with self.calls.scope("projection", self.implementation(module), site, {}) as call:
            call["inputs"] = self.calls.ports(input=input_)
            result = original(input_, *args, **kwargs)
            call["outputs"] = self.calls.ports(output=result[0], bias=result[1])
            return result

    def load_model(self, original, runner):
        from sglang.srt.layers.radix_attention import RadixAttention

        result = original(runner)
        for index, layer in enumerate(runner.model.model.layers):
            site = f"layers.{index}"
            layer.forward = partial(self.layer, layer.forward, layer, site)
            self_attention = layer.self_attn
            attention = self_attention.attn
            if not isinstance(attention, RadixAttention):
                raise ValueError(f"unsupported attention implementation at {site}")

            # Separate the producer and its native consumer so the compiler can retain their tensor edge.
            projection = self_attention.o_proj
            attention.forward = partial(self.attention, attention.forward, attention, f"{site}.self_attn.attn")
            projection.forward = partial(self.projection, projection.forward, projection, f"{site}.self_attn.o_proj")
        return result

    def install(self):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        scheduler = "sglang.srt.managers.scheduler.Scheduler."
        sender = "sglang.srt.managers.scheduler_components.output_sender.SenderWrapper.send_output"
        HookRegistry.register(scheduler + "flush_cache", self.flush, HookType.AROUND)
        HookRegistry.register(sender, self.publish, HookType.AROUND)
        if self.detailed:
            HookRegistry.register(scheduler + "run_batch", self.batch, HookType.AROUND)
            HookRegistry.register(scheduler + "process_batch_result", self.process_result, HookType.AROUND)
            HookRegistry.register("sglang.srt.model_executor.model_runner.ModelRunner.load_model",
                                  self.load_model, HookType.AROUND)


def install():
    path = os.environ.get("PROLOGUE_OBSERVATION")
    if path is not None:
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        Observation(path.parent, document["requests"], document["detailed"]).install()
