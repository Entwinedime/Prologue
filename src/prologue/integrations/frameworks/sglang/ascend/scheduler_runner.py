"""Execute one native scheduler cut with the logical TP group on the requested physical placement."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic_ns

from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.batches import BatchTensors
from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration
from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes
from prologue.integrations.frameworks.sglang.observation.profiles import NativeProfiles
from prologue.integrations.frameworks.sglang.observation.role_plan import NativeWorkerLayout
from prologue.integrations.platforms.ascend.markers import AscendMarkers


class WorkerProbes:
    def __init__(self, revision):
        self.revision = revision

    def run(self, request, output, mode):
        subprocess.run([sys.executable, "-m", "prologue.app.capture_worker", "capture", "--input", str(request),
                        "--output", str(output), mode], check=True)
        manifest = json.loads((output / "capture.json").read_bytes())
        if manifest["framework_revision"] != self.revision or manifest["engine_starts"]:
            raise ValueError("local probes must execute native worker cuts on the same framework")
        return manifest

    def pair(self, request, output):
        probes, starts = {}, 0

        # Independent processes keep structure instrumentation out of the CPU service measurement.
        for key, mode in (("structure", "--local-worker"), ("timing", "--local-host")):
            starts += self.run(request, output / key, mode)["model_runner_starts"]
            probes[key] = key
        write_json(output / "capture.json", {"framework_revision": self.revision, "engine_starts": 0,
                                             "model_runner_starts": starts, "probes": probes})

    def calibration(self, job, output):
        cuts = job.pop("calibration")
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output) as temporary:
            request = Path(temporary) / "job.json"
            write_json(request, job)
            primary = self.run(request, output, "--local-worker")
            paths = {}
            for name, batch in cuts.items():
                paths[name] = f"calibration/{name}"
                write_json(request, job | {"batch": batch})
                measured = self.run(request, output / paths[name], "--local-host-pair")
                primary["model_runner_starts"] += measured["model_runner_starts"]
        write_json(output / "capture.json", primary | {"calibration": paths})


class SchedulerRunner:
    def __init__(self, job, output):
        self.model = job["model"]
        self.configuration = NativeConfiguration(job["configuration"])
        self.batch = job.get("batch")
        self.staged = self.configuration.tp > 1 and job["single_device"]
        self.output = output

    def prepare_pools(self, scheduler, args):
        from sglang.benchmark import one_batch

        # The control loop starts with empty request and KV pools after one real native forward in each phase.
        runner = scheduler.tp_worker.model_runner
        reqs = one_batch.prepare_synthetic_inputs_for_latency_test(1, 1, [[0]])
        tokens, _, batch = one_batch.extend(reqs, runner)
        one_batch.decode(tokens, batch, runner)
        one_batch.synchronize(runner.device)
        if not scheduler.flush_cache(empty_cache=False):
            raise ValueError("native scheduler refused to reset its idle pools after the local warmup")
        one_batch.synchronize(runner.device)

        allocator, slots = scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool
        readback = {
            "logical_kv": "empty_after_native_flush", "free_tokens": allocator.available_size(),
            "free_slots": slots.available_size(),
            "workspace": {"prefill_tokens": [len(req.origin_input_ids) for req in batch.reqs], "decode_steps": 1,
                          "device_drained": True, "device_cache_emptied": False},
        }
        settings = {key: getattr(args, key) for key in self.configuration.settings()
                    if key not in {"prefill_chunk_tokens", "microbatch_requests"}}
        settings["prefill_chunk_tokens"] = scheduler.chunked_prefill_size or 0
        if "microbatch_requests" in self.configuration.settings():
            settings["microbatch_requests"] = args.pp_max_micro_batch_size
        return {"capacity": BatchTensors.capacity(allocator, slots), "configured_settings": settings, "initial_state": readback}

    def worker(self, rank, revision, args, ports, host_timing):
        import torch
        import torch.distributed as distributed
        from sglang.srt.distributed import get_world_group, parallel_state
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.plugins.hook_registry import HookRegistry

        from prologue.integrations.frameworks.sglang.ascend.batch_runner import SchedulerBatch, SchedulerBatchHostTiming, SchedulerBatchObservation

        world = self.configuration.tp
        directory = self.output if world == 1 else self.output / "workers" / str(rank)
        (directory / "runtime").mkdir(parents=True, exist_ok=True)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(directory / "runtime")
        device = 0 if self.staged else rank
        if torch.npu.device_count() != (1 if self.staged else world):
            raise ValueError("visible devices differ from the declared TP placement")
        torch.npu.set_device(device)

        model_cut = self.batch is not None and self.batch["cut"] == "model"
        description = {"mode": "null", "rank_offset": 0, "pp_size": args.pp_size, "ep_size": args.ep_size,
                       "worker_count": world, "instance_directory": str(self.output)}
        probe = SchedulerBatchHostTiming if host_timing else SchedulerBatchObservation
        observer = probe(directory, description, model=model_cut)
        started = monotonic_ns()
        if self.staged:
            from prologue.integrations.frameworks.sglang.ascend.shard_transport import ShardTransport

            ShardTransport.world(world, rank, (self.output / "rendezvous").as_uri())
            ShardTransport(self.output, *observer.staging()).install(description)
        observer.install()

        # Upstream applies all hooks for a target once. Install observation and transport together before model loading.
        HookRegistry.apply_hooks()
        scheduler = Scheduler(args, ports, gpu_id=device, tp_rank=rank, moe_ep_rank=0,
                              pp_rank=0, attn_cp_rank=0, moe_dp_rank=0, dp_rank=None)
        initialized = monotonic_ns()
        try:
            runner = scheduler.tp_worker.model_runner
            layout = NativeWorkerLayout(runner, revision, 0)
            layout.pools(scheduler, args)
            modules = dict(runner.model.named_modules())
            _, experts = ModelScopes.expert_structure(modules)
            if experts:
                raise ValueError("this scheduler cut requires a dense model")
            roles = layout.roles(ModelScopes.selected(modules), experts, ()).document(None)
            environment = layout.environment(scheduler, roles["workers"], AscendMarkers.environment())
            write_json(directory / "binding.json", self.prepare_pools(scheduler, args))
            observer.bind(scheduler, layout, environment)
            prepared = monotonic_ns()

            observed = roles["observed_roles"]
            if world > 1 and self.batch is not None:
                observed = self.batch.get("peer_ranks", [self.batch["rank"]])
                if "peer_ranks" in self.batch and self.staged:
                    raise ValueError("peer service requires simultaneous observation on distinct physical devices")
            for window in NativeProfiles.windows(self.output, observed, self.staged, 1):
                selected = rank in window.observed
                if world > 1:
                    distributed.barrier(group=get_world_group().cpu_group)
                cut = SchedulerBatch(scheduler, observer, model=model_cut, observed=selected)
                boundary = cut.observe(ports, self.batch, warm_cost=host_timing)
                if selected:
                    observer.export()
                    write_json(directory / "boundary.json", boundary)
                if world > 1:
                    distributed.barrier(group=get_world_group().cpu_group)
        finally:
            scheduler.release_host_resources()
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()

        write_json(directory / "capture.json", {
            "framework_revision": revision, "rank": rank, "observed": rank in observed,
            "binding": "binding.json", "boundary": "boundary.json", **({"host": "host.json"} if host_timing else {"facts": "facts.json"}),
            "engine_starts": 0, "model_runner_starts": 1, "initialization_ns": initialized - started,
            "preparation_and_readback_ns": prepared - initialized, "capture_and_export_ns": monotonic_ns() - prepared,
            **({"timing_warmup": cut.preparation} if cut.preparation is not None else {}),
        })

    def run(self, revision, *, host_timing=False):
        from sglang.benchmark import one_batch
        from torch.multiprocessing import spawn

        args = one_batch.ServerArgs(**self.configuration.runner_options(self.model))
        one_batch._set_envs_and_config(args)
        ports = one_batch.PortArgs.init_new(args)
        world = self.configuration.tp
        inputs = (revision, args, ports, host_timing)
        if world == 1:
            self.worker(0, *inputs)
            return

        # Every rank receives the same native group and ports, including peers that only supply operands.
        spawn(self.worker, args=inputs, nprocs=world, join=True)
        workers = {}
        for rank in range(world):
            relative = f"workers/{rank}"
            manifest = json.loads((self.output / relative / "capture.json").read_bytes())
            if manifest["observed"]:
                workers[str(rank)] = relative
        write_json(self.output / "capture.json", {"framework_revision": revision, "workers": workers,
                   "native_participants": list(range(world)), "engine_starts": 0, "model_runner_starts": world})
