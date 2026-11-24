"""Sample Decode attention on a native ModelRunner and its actual KV pool."""

import json
import os
import sys
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.acquisition.observation.calls import Calls
from prologue.integrations.frameworks.sglang.configuration import Configuration, Model, native_options
from prologue.integrations.frameworks.sglang.observation.attention import AttentionCall, implementation
from prologue.integrations.platforms.ascend.profile import task_profile


class DecodeAttention:
    mode = "decode-attention"

    @staticmethod
    def length(domain, coordinate):
        config = domain["configuration"]
        geometry = coordinate["geometry"]
        if (domain["hardware"]["platform"] != "ascend" or config.get("tp") != 1 or config.get("pp", 1) != 1
                or config["attention_backend"] != "ascend" or geometry.get("phase") != "decode"
                or coordinate["family"] != "device_task" or coordinate.get("operation") != "attention"
                or len(geometry.get("sequence_lengths", ())) != 1):
            return None

        length, = geometry["sequence_lengths"]
        page = config["page_size"]
        if type(length) is not int or not 1 < length <= config["max_prefill_tokens"] + 1:
            return None
        pages = (length + page - 1) // page
        if pages * page > config["max_total_tokens"]:
            return None
        return length

    def __init__(self, request, output):
        self.started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)

        from sglang.benchmark import one_batch

        self.domain = request["domain"]
        self.output = output
        self.sequences = request["sequence_lengths"]
        self.site_type = request["implementation"]
        self.native = one_batch
        if os.environ["PROLOGUE_SGLANG_REVISION"] != self.domain["framework_revision"]:
            raise ValueError("the sampler and the requested work require the same native framework revision")

        config = Configuration(**self.domain["configuration"])
        model = Model(**self.domain["model"])
        self.args = one_batch.ServerArgs(**native_options(model, config))

    def initialize(self):
        native = self.native
        native._set_envs_and_config(self.args)
        native.initialize_moe_config(self.args)
        native.initialize_fp8_gemm_config(self.args)
        native.initialize_fp4_gemm_config(self.args)
        runner, _ = native.load_model(self.args, native.PortArgs.init_new(self.args), gpu_id=0, tp_rank=0)
        self.runner = runner
        self.site = next(((name, module) for name, module in runner.torch_runner.model.named_modules()
                          if implementation(module) == self.site_type), None)

    def prepare(self, length):
        self.runner.clear()
        reqs = self.native.prepare_synthetic_inputs_for_latency_test(1, length - 1, [[1] * (length - 1)])
        tokens, _, batch = self.runner.extend(reqs)
        self.runner.synchronize()
        return tokens, batch

    def warm(self):
        for length in self.sequences:
            tokens, batch = self.prepare(length)
            self.runner.decode(tokens, batch)
            self.runner.synchronize()

    def observe(self, length):
        from torch.autograd.profiler import record_function

        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        tokens, batch = self.prepare(length)
        directory = self.output / str(length)
        directory.mkdir()
        name, module = self.site
        calls = Calls(record_function)
        call = AttentionCall(calls, module, name, require_parent=False)
        module.forward = call
        try:
            with task_profile(directory / "profile"):
                began = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
                self.runner.decode(tokens, batch)
                self.runner.synchronize()
                ended = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        finally:
            module.forward = call.forward
        exported = clock_gettime_ns(CLOCK_MONOTONIC_RAW)

        slot, = batch.req_pool_indices_cpu.tolist()
        request, = batch.reqs
        observed, = batch.seq_lens_cpu.tolist()
        if (len(calls.rows) != 1 or calls.rows[0]["parameters"]["seq_lens"] != [length]
                or calls.rows[0]["parameters"]["phase"] != "decode" or observed != length
                or request.kv_committed_len != length):
            return {"sequence_length": length, "status": "untriggered", "observed_length": observed}

        # Read the committed table after trace closure so the evidence fetch cannot change measured work.
        locations = batch.req_to_token_pool.req_to_token[slot, :length].tolist()
        written, = batch.out_cache_loc.tolist()
        if locations[-1] != written or min(locations) <= 0 or len(set(locations)) != length:
            raise ValueError("native Decode did not retain a valid context and its appended KV slot")
        calls.rows[0]["members"] = [{"request": str(request.rid), "slot": slot,
                                     "kv_committed_tokens": request.kv_committed_len}]
        worker = {"calls": calls.rows, "publications": {},
                  "kv": {"slot": slot, "locations": locations, "write": written}}
        (directory / "worker.json").write_text(json.dumps(worker) + "\n", encoding="utf-8")
        trace, = (directory / "profile").rglob("trace_view.json")
        manifest = {key: self.domain[key] for key in ("framework_revision", "configuration", "model")}
        manifest.update(mode=self.mode, trace=str(trace.relative_to(directory)),
                        delivery={"origin_raw_ns": began}, profile_export_ns=exported - ended)
        (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return {"sequence_length": length, "status": "observed", "observation": directory.name,
                "prepare_ns": began - started, "observe_ns": ended - began, "export_ns": exported - ended}

    def run(self):
        self.initialize()
        initialized = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        samples = []
        warmed = initialized
        if self.site is not None:
            self.warm()
            warmed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            for length in self.sequences:
                samples.append(self.observe(length))
            self.runner.clear()

        # This document seals the completed native attempt for recovery in the workflow process.
        final = self.output / "native.json"
        pending = final.with_suffix(".tmp")
        pending.write_text(json.dumps({
            "status": "observed" if any(row["status"] == "observed" for row in samples) else "untriggered",
            "model_runner_starts": 1, "engine_starts": 0, "initialization_ns": initialized - self.started,
            "warmup_ns": warmed - initialized, "samples": samples,
        }) + "\n", encoding="utf-8")
        pending.replace(final)


if __name__ == "__main__":
    plan_path, out = map(Path, sys.argv[1:])
    DecodeAttention(json.loads(plan_path.read_text(encoding="utf-8")), out).run()
