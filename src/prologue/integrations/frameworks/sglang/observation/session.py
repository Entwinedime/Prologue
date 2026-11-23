"""Run the native request lifecycle and close one observation window."""

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.integrations.frameworks.sglang.configuration import Configuration, Model, native_options
from prologue.integrations.frameworks.sglang.observation.parallel import RankPlan


class Session:
    def __init__(self, receipt, configuration, workload, output, mode, placement):
        self.model = Model(**receipt["model"])
        configured = next(row for row in receipt["configurations"] if row["id"] == configuration)
        self.configuration = Configuration(**configured)
        self.workload = next(row for row in receipt["workloads"] if row["id"] == workload)
        self.output = output.resolve()
        self.mode = mode
        self.placement = placement
        if self.configuration.tp > 1 and mode != "detailed":
            raise ValueError("native TP groups currently require detailed observation")

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

    def aliases(self, phase):
        return {f"{phase}/{index}": request["id"] for index, request in enumerate(self.workload["requests"])}

    def record(self, engine, directory, phase):
        detail = self.mode == "detailed"
        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        if detail:
            # The semantic scopes carry call identity and tensor layout.
            engine.start_profile(output_dir=str(directory), activities=["CPU", "GPU"],
                                 with_stack=False, record_shapes=False)
        try:
            delivery = engine.loop.run_until_complete(self.dispatch(engine, phase))
            self.drain(engine)
            observed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        finally:
            if detail:
                engine.stop_profile()
        record = {"delivery": delivery, "observation_ns": observed - started}
        if detail:
            record["profile_export_ns"] = clock_gettime_ns(CLOCK_MONOTONIC_RAW) - observed
        return record

    def check_publications(self, directory):
        worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        for request in self.workload["requests"]:
            rows = worker["publications"][request["id"]]
            if not rows or rows[-1]["count"] != request["output_tokens"]:
                raise ValueError(f"incomplete native publication evidence for {request['id']}")

    def record_workers(self, engine):
        if self.configuration.tp == 1:
            recorded = self.record(engine, self.output / "profile", "source")
            self.check_publications(self.output)
            if self.mode == "detailed":
                trace, = (self.output / "profile").rglob("trace_view.json")
                recorded["trace"] = str(trace.relative_to(self.output))
            return recorded

        plan = RankPlan(self.output, self.configuration.tp, self.placement)
        windows = []
        for index, ranks in enumerate(plan.windows):
            directory = self.output / f"window-{index}"
            directory.mkdir()
            phase = f"source-{index}"
            selection = {"ranks": ranks, "requests": self.aliases(phase)}
            (directory / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            record = self.record(engine, directory, phase)
            self.check_publications(directory / "rank-0")

            traces = {}
            for rank in ranks:
                trace, = (directory / f"rank-{rank}" / "profile").rglob("trace_view.json")
                traces[rank] = str(trace.relative_to(self.output))
            windows.append({**record, "directory": directory.name, "traces": traces})
        return {"workers": plan.workers, "windows": windows}

    def run(self):
        self.output.mkdir()
        parallel = self.configuration.tp > 1
        specification = self.output / "observation.json"
        description = {"requests": self.aliases("source"), "mode": self.mode,
                       "parallel_placement": self.placement if parallel else None}
        specification.write_text(json.dumps(description), encoding="utf-8")
        os.environ["PROLOGUE_OBSERVATION"] = str(specification)

        if parallel and self.placement == "single_device":
            from prologue.integrations.frameworks.sglang.ascend.shards import SingleDeviceEngine as Engine
        else:
            from sglang import Engine

        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        with Engine(**native_options(self.model, self.configuration)) as engine:
            initialized = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            engine.loop.run_until_complete(self.dispatch(engine, "warmup"))
            self.drain(engine)
            warmed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            recorded = self.record_workers(engine)

        manifest = {
            "mode": self.mode, "framework_revision": os.environ["PROLOGUE_SGLANG_REVISION"],
            "model": asdict(self.model), "configuration": asdict(self.configuration), "workload": self.workload,
            "initial_state": {"kv": "empty", "workspace": "warmed"}, "placement": self.placement,
            "clock": "monotonic_raw_ns", **recorded,
            "initialization_ns": initialized - started, "warmup_ns": warmed - initialized,
        }
        manifest["model_config"] = json.loads((Path(self.model.path) / "config.json").read_text(encoding="utf-8"))
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
