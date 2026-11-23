"""Own one native Engine and its observation lifecycle within a capture job."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from prologue.integrations.frameworks.sglang.configuration import CaptureInput, InstanceConfiguration, NativeConfiguration
from prologue.integrations.frameworks.sglang.ingress import RequestSubmissions
from prologue.integrations.frameworks.sglang.observation.profiles import NativeProfiles
from prologue.integrations.frameworks.sglang.observation.submissions import EngineSubmissions
from prologue.integrations.frameworks.sglang.observation.window import ObservationWindow
from prologue.artifacts import write_json


class NativeServingInstance:
    def __init__(
        self, source: CaptureInput, instance: InstanceConfiguration, output: Path, index: int, rank_offset: int,
        *, resource_trace: bool, latency_only: bool, host_timing: bool, service_reference: bool, local_shards: bool,
        attention_sites=None,
    ) -> None:
        self.model = source.model
        self.observe_all_workers = source.observe_all_workers or service_reference
        self.resource_trace = resource_trace
        self.latency_only = latency_only
        self.host_timing = host_timing
        self.attention_sites = attention_sites
        self.service_reference = service_reference
        self.local_shards = local_shards
        self.output = output.resolve()
        self.instance = instance
        self.instance_count, self.worker_count = len(source.instances), source.worker_count
        self.directory = self.output / "instances" / str(index)
        self.rank_offset = rank_offset
        configuration = instance.configuration
        self.configuration = configuration.values
        self.workload = source.workload
        self.submission_groups = source.submission_groups
        self.tp, self.pp, self.ep = configuration.tp, configuration.pp, configuration.ep
        self.roles = tuple(range(rank_offset, rank_offset + self.tp * self.pp))
        self.delivery = {}

    def generate(self, engine: Any, phase: str, submissions: EngineSubmissions, clock: Any, barrier: Any) -> None:
        submissions.ingress.content.check_vocabulary(engine.tokenizer_manager.model_config.vocab_size)
        print(f"prologue: native {phase} begin", flush=True)
        try:
            engine.loop.run_until_complete(submissions.run(engine, clock, barrier, self.output, self.roles))
        finally:

            # Delivery records explain actual ingress delay; declared arrival_ns is never rewritten.
            self.delivery[phase] = submissions.delivery
            write_json(self.directory / "request-delivery.json", self.delivery)
        print(f"prologue: native {phase} complete", flush=True)

    def reset(self, engine: Any, phase: str) -> None:

        # End observed serving at native idle before sending the driver's administrative reset.
        ObservationWindow.wait(self.output, self.roles, phase, "drained")
        result = engine.loop.run_until_complete(engine.tokenizer_manager.flush_cache(timeout_s=30))
        if not result.success:
            raise RuntimeError(f"native cache reset failed: {result.message}")
        ObservationWindow.wait(self.output, self.roles, phase, "complete")

    def timing_metrics(self):
        records = {str(rank): self.directory / "timing" / f"rank-{rank}" / "record.json" for rank in self.roles}
        if self.host_timing:
            return {"host_samples": {rank: str(path.relative_to(self.output)) for rank, path in records.items()}}
        if self.attention_sites:
            return {"attention_samples": {rank: str(path.relative_to(self.output)) for rank, path in records.items()}}

        measurements = [json.loads(path.read_text()) for path in records.values()]
        requests = {request["id"]: {"id": request["id"]} for request in self.workload["requests"]}
        for record in measurements:
            for row in record["requests"]:
                combined = requests[row["id"]]
                values = {name: value for name, value in row.items() if name != "id"}
                if values.keys() & combined.keys():
                    raise ValueError(f"more than one worker measured a request boundary for {row['id']}")
                combined.update(values)
        required = {"id", "ingress_raw_ns", "first_publication_raw_ns", "last_publication_raw_ns", "output_tokens"}
        if any(row.keys() != required for row in requests.values()):
            raise ValueError("native request measurement did not join ingress, first token and final token")
        loads = [{"rank": record["rank"], **event} for record in measurements for event in record["specialization_loads"]]
        path = self.directory / "latency.json"
        write_json(path, {"clock": "monotonic_raw_ns", "requests": list(requests.values()), "specialization_loads": loads}, replace=False)
        return {"latency": str(path.relative_to(self.output))}

    def observe(self, engine, capture, clock, barrier):
        windows = None
        round_index = 0
        while True:
            engine.start_profile(output_dir=str(self.output / "native"), activities=["CPU", "GPU"],
                                 record_shapes=False, with_stack=False)
            try:
                ObservationWindow.wait(self.output, self.roles, "profiler", "complete")
                if windows is None:
                    plan = json.loads((self.directory / "role-plan.json").read_text())
                    windows = NativeProfiles.windows(self.output, plan["observed_roles"], self.local_shards, self.instance_count)
                window = windows[round_index]
                barrier.wait()
                phase = f"capture/{window.observed[0]}" if window.replay else "capture"
                self.generate(engine, phase, capture, clock, barrier)
                self.reset(engine, "source")
                barrier.wait()
            finally:
                engine.stop_profile()
            participants = range(self.worker_count) if window.replay else self.roles
            NativeProfiles.wait(self.output, window.observed, participants, window.protocol_directory(self.output))
            round_index += 1
            if round_index == len(windows):
                replays = [item.observed for item in windows if item.replay and item.observed[0] in self.roles]
                return plan["observed_roles"], replays

    def run(self, barrier, clock, bootstrap, rendezvous) -> dict:
        self.directory.mkdir()
        description = self.directory / "session.json"
        mode = self.instance.mode
        capture = EngineSubmissions(self.workload, mode, self.submission_groups)
        preparation = {}
        if any(len(arrivals) > 1 for arrivals in capture.groups):

            # Cold compilation can overlap the next arrival. Replay the paced workload once warm before measuring it.
            preparation["prime"] = EngineSubmissions(self.workload, mode, self.submission_groups)
        preparation["warmup"] = EngineSubmissions(self.workload, mode, self.submission_groups)
        passes = {**preparation, "source": capture}
        count = len(self.workload["requests"])

        # PD bootstrap rooms must stay distinct across the drain and logical cache reset.
        for ordinal, (phase, submissions) in enumerate(passes.items()):
            submissions.prepare(RequestSubmissions.observed if phase == "source" else phase, bootstrap, 1 + ordinal * count)
        session = {
            "output": str(self.output), "instance_directory": str(self.directory), "mode": mode,
            "rendezvous": {"host": rendezvous["MASTER_ADDR"], "port": int(rendezvous["MASTER_PORT"])},
            "rank_offset": self.rank_offset, "native_participants": self.roles,
            "pp_size": self.pp, "ep_size": self.ep,
            "observe_all_workers": self.observe_all_workers, "local_shards": self.local_shards,
            "instance_count": self.instance_count, "worker_count": self.worker_count,
            "resource_trace": self.resource_trace,
            "latency_only": self.latency_only,
            "host_timing": self.host_timing,
            "service_reference": self.service_reference,
            **{f"{phase}_requests": [RequestSubmissions.native_id(submissions.prefix, index) for index in range(count)]
               for phase, submissions in passes.items()},
            "representative_warmup": True,
            "protocol_requests": capture.protocol_requests,
            "content": self.workload,
            "request_groups": {phase: submissions.request_groups() for phase, submissions in passes.items()},
            "serving_input": {"configuration": self.configuration, **self.workload},
        }
        if self.attention_sites:
            session["attention_sites"] = self.attention_sites
        write_json(description, session)
        os.environ.update(rendezvous)
        os.environ["PROLOGUE_GRAPH_SESSION"] = str(description)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(self.output / "runtime")
        os.environ["SGLANG_PLUGINS"] = "prologue"
        if self.ep > 1:
            os.environ.update(DEEP_USE_ALLTOALL_MODE="1", DEEPEP_HCCL_BUFFSIZE="1024", HCCL_BUFFSIZE="1024")

        if self.local_shards:
            from prologue.integrations.frameworks.sglang.shard_engine import SingleDeviceEngine as Engine
        else:
            from sglang.srt.entrypoints.engine import Engine

        model = self.model
        configuration = self.configuration
        chunk_tokens = configuration["prefill_chunk_tokens"]
        options = {
            "model_path": model["path"], "dtype": model["dtype"], "device": NativeConfiguration.accelerator_device,
            "tp_size": self.tp, "pp_size": self.pp, "base_gpu_id": 0 if self.local_shards else self.rank_offset,
            "nccl_port": int(rendezvous["MASTER_PORT"]),
            "attention_backend": configuration["attention_backend"],
            "page_size": configuration["page_size"],
            "max_running_requests": configuration["max_running_requests"],
            "max_total_tokens": configuration["max_total_tokens"],
            "max_prefill_tokens": configuration["max_prefill_tokens"],
            "chunked_prefill_size": chunk_tokens if chunk_tokens > 0 else -1,
            "enable_mixed_chunk": configuration["mixed_batches"],
            "disable_radix_cache": not configuration["prefix_cache"],
            "disable_overlap_schedule": True, "disable_cuda_graph": True,
            "skip_tokenizer_init": True, "load_format": "dummy", "sleep_on_idle": True, "trust_remote_code": True,
            "mem_fraction_static": 0.8, "log_level": "warning",
        }
        if mode != "null":
            options.update(
                disaggregation_mode=mode, disaggregation_transfer_backend="ascend",
                disaggregation_bootstrap_port=bootstrap["port"],
                disaggregation_decode_enable_radix_cache=mode == "decode" and configuration["prefix_cache"],
            )
        if "microbatch_requests" in configuration:
            options["pp_max_micro_batch_size"] = configuration["microbatch_requests"]
        if self.ep > 1:
            options.update(ep_size=self.ep, moe_a2a_backend="deepep", deepep_mode="normal")
        from sglang.srt.plugins import load_plugins

        # Native argument normalization can change EP size; reject that change before loading weights or launching workers.
        load_plugins()
        args = Engine.server_args_class(**options)
        if (args.tp_size, args.pp_size, args.ep_size) != (self.tp, self.pp, self.ep):
            raise ValueError("native configuration changed the requested parallel topology")
        started_ns = time.monotonic_ns()
        engine = Engine(server_args=args)
        try:
            initialized_ns = time.monotonic_ns()
            for phase, submissions in preparation.items():
                barrier.wait()
                self.generate(engine, phase, submissions, clock, barrier)
                self.reset(engine, phase)
            warmed_ns = time.monotonic_ns()

            barrier.wait()
            metrics = {}
            if self.latency_only:
                self.generate(engine, "capture", capture, clock, barrier)
                self.reset(engine, "source")
                selected = ()
                metrics = self.timing_metrics()
            else:
                selected, windows = self.observe(engine, capture, clock, barrier)
                if self.local_shards:
                    metrics["profile_windows"] = windows
            captured_ns = time.monotonic_ns()
        finally:
            try:
                engine.shutdown()
            finally:

                # Engine stops its workers but leaves the Prefill bootstrap thread to its owner.
                if engine.tokenizer_manager.bootstrap_server is not None:
                    engine.tokenizer_manager.bootstrap_server.close()
        return {
            "id": self.instance.identity, "mode": mode, "rank_offset": self.rank_offset, "observed_roles": selected,
            "native_participants": self.roles, **({"placement": "single_device"} if self.local_shards else {}),
            "initialization_ns": initialized_ns - started_ns,
            "warmup_ns": warmed_ns - initialized_ns,
            "capture_ns": captured_ns - warmed_ns, **metrics,
        }
