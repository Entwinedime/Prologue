"""Run the native request lifecycle and close one observation window."""

import asyncio
import json
import os
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns, monotonic, sleep

from prologue.integrations.frameworks.sglang.configuration import native_options
from prologue.integrations.frameworks.sglang.content import PromptContent
from prologue.integrations.frameworks.sglang.observation.parallel import RankPlan


class Session:
    def __init__(self, model, configuration, workload, output, mode, placement):
        self.model = model
        self.configuration = configuration
        self.workload = workload
        self.output = output.resolve()
        self.mode = mode
        self.placement = placement
        self.content = PromptContent.declared(workload, configuration.prefix_cache)
        self.ranked = configuration.worker_count > 1
        if self.ranked and mode != "detailed":
            raise ValueError("parallel workers currently require detailed observation")

    def output_count(self, request):
        return request["output_tokens"]

    async def generate(self, engine, phase, requests, **transport):
        identities = [f"{phase}/{index}" for index, _ in requests]
        prompts = [self.content.prompt(request["id"]) if self.content is not None else [1] * request["prompt_tokens"]
                   for _, request in requests]
        sampling = [{"max_new_tokens": request["output_tokens"], "ignore_eos": True, "temperature": 0}
                    for _, request in requests]
        submitted = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        output = await engine.async_generate(
            rid=identities, input_ids=prompts, stream=True, sampling_params=sampling, **transport,
        )
        finished = {}
        async for chunk in output:
            metadata = chunk["meta_info"]
            if metadata["finish_reason"] is not None:
                finished[metadata["id"]] = metadata["completion_tokens"]
        for identity, (_, request) in zip(identities, requests, strict=True):
            if finished.get(identity) != self.output_count(request):
                raise ValueError(f"native output count did not match request {request['id']}")
        return [{"request": request["id"], "submitted_raw_ns": submitted, "offset_ns": request["arrival_ns"]}
                for _, request in requests]

    async def dispatch(self, engine, phase):
        return await self.dispatch_at(engine, phase, clock_gettime_ns(CLOCK_MONOTONIC_RAW))

    async def dispatch_at(self, engine, phase, origin, *role_arguments):
        pending = []
        for offset, tied in groupby(enumerate(self.workload["requests"]), key=lambda item: item[1]["arrival_ns"]):

            # One native request envelope preserves tied ingress without deciding serving batches.
            delay_ns = origin + offset - clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            pending.append(asyncio.create_task(self.generate(engine, phase, list(tied), *role_arguments)))
        delivered = [row for cohort in await asyncio.gather(*pending) for row in cohort]
        return {"origin_raw_ns": origin, "requests": delivered}

    @staticmethod
    def drain(engine):
        result = engine.loop.run_until_complete(engine.tokenizer_manager.flush_cache(timeout_s=30))
        if not result.success:
            raise RuntimeError(f"native drain failed: {result.message}")

    def aliases(self, phase):
        return {f"{phase}/{index}": request["id"] for index, request in enumerate(self.workload["requests"])}

    def await_profiles(self, engine, directory, marker):
        if not self.ranked:
            return

        # The tokenizer receives one reply even when the native pipeline propagates the request farther.
        missing = {directory / f"rank-{rank}" / marker for rank in range(self.configuration.worker_count)}
        deadline = monotonic() + engine.server_args.watchdog_timeout
        while missing:
            missing = {path for path in missing if not path.is_file()}
            if missing:
                if monotonic() >= deadline:
                    ranks = ", ".join(sorted(path.parent.name for path in missing))
                    raise TimeoutError(f"native profile {marker} missing acknowledgments from {ranks}")
                sleep(0.02)

    def record(self, engine, directory, phase):
        detail = self.mode == "detailed"
        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        if detail:
            # The semantic scopes carry call identity and tensor layout.
            engine.start_profile(output_dir=str(directory), activities=["CPU", "GPU"],
                                 with_stack=False, record_shapes=False)
        try:
            if detail:
                self.await_profiles(engine, directory, "started")
            delivery = engine.loop.run_until_complete(self.dispatch(engine, phase))
            self.drain(engine)
            observed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        finally:
            if detail:
                engine.stop_profile()
                self.await_profiles(engine, directory, "stopped")
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
        if not self.ranked:
            recorded = self.record(engine, self.output / "profile", "source")
            self.check_publications(self.output)
            if self.mode == "detailed":
                trace, = (self.output / "profile").rglob("trace_view.json")
                recorded["trace"] = str(trace.relative_to(self.output))
            return recorded

        plan = RankPlan(self.output, self.configuration, self.placement)
        windows = []
        for index, ranks in enumerate(plan.windows):
            windows.append(self.record_ranks(engine, index, ranks, plan.publisher))
        return {"workers": plan.workers, "windows": windows}

    def record_ranks(self, engine, index, ranks, publisher):
        directory = self.output / f"window-{index}"
        directory.mkdir()
        phase = f"source-{index}"
        selection = {"ranks": ranks, "requests": self.aliases(phase)}
        (directory / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
        record = self.record(engine, directory, phase)
        self.check_publications(directory / f"rank-{publisher}")

        traces = {}
        for rank in ranks:
            trace, = (directory / f"rank-{rank}" / "profile").rglob("trace_view.json")
            traces[rank] = str(trace.relative_to(self.output))
        return {**record, "directory": directory.name, "traces": traces}

    def prepare(self, **details):
        self.output.mkdir()
        specification = self.output / "observation.json"
        config = self.configuration
        description = {"requests": self.aliases("source"), "mode": self.mode, "pp": config.pp,
                       "snapshot_cache": bool(config.prefill_chunk_tokens or config.prefix_cache),
                       "prefix_cache": config.prefix_cache,
                       "parallel_placement": self.placement if self.ranked else None, **details}
        if self.content is not None:
            description["synthetic_sampling"] = {"outputs": self.content.outputs, "warmup": self.aliases("warmup"),
                                                 "vocabulary": self.content.vocabulary}
        ep = config.ep
        if ep > 1:
            os.environ.update(DEEP_USE_ALLTOALL_MODE="1", DEEPEP_HCCL_BUFFSIZE="1024", HCCL_BUFFSIZE="1024")
        routes = config.route_table(self.workload)
        if routes is not None:
            description["experts"] = {"width": ep, "routes": routes, "requests": self.workload["requests"],
                                      "warmup": self.aliases("warmup")}
        specification.write_text(json.dumps(description), encoding="utf-8")
        os.environ["PROLOGUE_OBSERVATION"] = str(specification)

        if self.ranked and self.placement == "single_device":
            from prologue.integrations.frameworks.sglang.ascend.shards import SingleDeviceEngine as Engine
        else:
            from sglang import Engine

        return Engine

    def run(self):
        Engine = self.prepare()
        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        with Engine(**native_options(self.model, self.configuration)) as engine:
            initialized = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            engine.loop.run_until_complete(self.dispatch(engine, "warmup"))
            self.drain(engine)
            warmed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            recorded = self.record_workers(engine)
        self.save(started, initialized, warmed, recorded)

    def save(self, started, initialized, warmed, recorded):
        manifest = {
            "mode": self.mode, "framework_revision": os.environ["PROLOGUE_SGLANG_REVISION"],
            "model": asdict(self.model), "configuration": self.configuration.document(), "workload": self.workload,
            "initial_state": {"kv": "empty", "workspace": "warmed"}, "placement": self.placement,
            "clock": "monotonic_raw_ns", **recorded,
            "initialization_ns": initialized - started, "warmup_ns": warmed - initialized,
        }
        manifest["model_config"] = json.loads((Path(self.model.path) / "config.json").read_text(encoding="utf-8"))
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
