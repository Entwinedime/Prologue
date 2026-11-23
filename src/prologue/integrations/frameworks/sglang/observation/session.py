"""Run the native request lifecycle and close one observation window."""

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.integrations.frameworks.sglang.configuration import Configuration, Model


class Session:
    def __init__(self, receipt, configuration, workload, output, mode):
        self.model = Model(**receipt["model"])
        self.configuration = Configuration(**next(row for row in receipt["configurations"] if row["id"] == configuration))
        self.workload = next(row for row in receipt["workloads"] if row["id"] == workload)
        self.output = output.resolve()
        self.mode = mode
        if self.configuration.tp != 1:
            raise ValueError("this observation entry supports a single native TP1 worker")

    def engine_options(self):
        configuration = asdict(self.configuration)
        del configuration["id"]
        configuration["tp_size"] = configuration.pop("tp")
        model = asdict(self.model)
        model["model_path"] = model.pop("path")
        return {
            **model, **configuration, "device": "npu", "skip_tokenizer_init": True,
            "disable_overlap_schedule": True, "disable_cuda_graph": True,
            "disable_radix_cache": True, "chunked_prefill_size": -1, "stream_interval": 1,
            "mem_fraction_static": 0.8, "log_level": "warning",
        }

    @staticmethod
    async def request(engine, native, request):
        tokens = [1] * request["prompt_tokens"]
        submitted = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        output = await engine.async_generate(
            rid=native, input_ids=tokens, stream=True,
            sampling_params={"max_new_tokens": request["output_tokens"], "ignore_eos": True, "temperature": 0},
        )
        last = None
        async for chunk in output:
            last = chunk
        if last is None or last["meta_info"]["completion_tokens"] != request["output_tokens"]:
            raise ValueError(f"native output count did not match request {request['id']}")
        return {"request": request["id"], "submitted_raw_ns": submitted, "offset_ns": request["arrival_ns"]}

    async def dispatch(self, engine, phase):
        origin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        pending = []
        for index, request in enumerate(self.workload["requests"]):
            # Submit tied requests in input order while earlier requests keep running.
            delay_ns = origin + request["arrival_ns"] - clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            pending.append(asyncio.create_task(self.request(engine, f"{phase}/{index}", request)))
        return {"origin_raw_ns": origin, "requests": await asyncio.gather(*pending)}

    @staticmethod
    def drain(engine):
        result = engine.loop.run_until_complete(engine.tokenizer_manager.flush_cache(timeout_s=30))
        if not result.success:
            raise RuntimeError(f"native drain failed: {result.message}")

    def run(self):
        self.output.mkdir()
        detail = self.mode == "detailed"
        specification = self.output / "observation.json"
        aliases = {f"source/{index}": request["id"] for index, request in enumerate(self.workload["requests"])}
        specification.write_text(json.dumps({"requests": aliases, "mode": self.mode}), encoding="utf-8")
        os.environ["PROLOGUE_OBSERVATION"] = str(specification)

        from sglang import Engine

        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        with Engine(**self.engine_options()) as engine:
            initialized = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            engine.loop.run_until_complete(self.dispatch(engine, "warmup"))
            self.drain(engine)
            warmed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)

            if detail:
                # Semantic scopes already record call identity and tensor layout.
                engine.start_profile(output_dir=str(self.output / "profile"), activities=["CPU", "GPU"],
                                     with_stack=False, record_shapes=False)
            try:
                delivery = engine.loop.run_until_complete(self.dispatch(engine, "source"))
                self.drain(engine)
                observed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            finally:
                if detail:
                    engine.stop_profile()
            exported = clock_gettime_ns(CLOCK_MONOTONIC_RAW)

        worker = json.loads((self.output / "worker.json").read_text(encoding="utf-8"))
        for request in self.workload["requests"]:
            rows = worker["publications"][request["id"]]
            if not rows or rows[-1]["count"] != request["output_tokens"]:
                raise ValueError(f"incomplete native publication evidence for {request['id']}")

        manifest = {
            "mode": self.mode, "framework_revision": os.environ["PROLOGUE_SGLANG_REVISION"],
            "model": asdict(self.model), "configuration": asdict(self.configuration), "workload": self.workload,
            "initial_state": {"kv": "empty", "workspace": "warmed"},
            "clock": "monotonic_raw_ns", "delivery": delivery,
            "initialization_ns": initialized - started, "warmup_ns": warmed - initialized,
            "observation_ns": observed - warmed,
        }
        manifest["model_config"] = json.loads((Path(self.model.path) / "config.json").read_text(encoding="utf-8"))
        if detail:
            trace, = (self.output / "profile").rglob("trace_view.json")
            manifest["trace"] = str(trace.relative_to(self.output))
            manifest["profile_export_ns"] = exported - observed
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
