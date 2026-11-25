"""Capture, export or interpret one native job from the worker CLI."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import socket
import time
import traceback
from contextlib import contextmanager
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain, route
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.window import (
    EngineSubmissions,
    NativeProfiles,
    ObservationWindow,
)
from prologue.integrations.frameworks.sglang.shared.batches import RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.configuration import (
    CaptureInput,
    InstanceConfiguration,
    NativeConfiguration,
)


class NativeServingInstance:
    def __init__(self, source: CaptureInput, instance: InstanceConfiguration, output: Path, index: int, options):
        self.model = source.model
        self.options = options
        self.local_shards = options["local_shards"]
        self.output = output.resolve()
        self.instance = instance
        self.instance_count, self.worker_count = len(source.instances), source.worker_count
        self.directory = self.output / "instances" / str(index)
        self.rank_offset = instance.ranks.start
        configuration = instance.configuration
        self.configuration = configuration.values
        self.workload = source.workload
        self.submission_groups = source.submission_groups
        self.tp, self.pp, self.ep = configuration.tp, configuration.pp, configuration.ep
        self.roles = tuple(instance.ranks)
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
        measured = submissions.prefix == RequestSubmissions.observed
        self.reset(engine, "source" if measured else phase)

    def reset(self, engine: Any, phase: str) -> None:
        # End observed serving at native idle before sending the driver's administrative reset.
        ObservationWindow.wait(self.output, self.roles, phase, "drained")
        result = engine.loop.run_until_complete(engine.tokenizer_manager.flush_cache(timeout_s=30))
        if not result.success:
            raise RuntimeError(f"native cache reset failed: {result.message}")
        ObservationWindow.wait(self.output, self.roles, phase, "complete")

    def timing_metrics(self):
        records = {str(rank): self.directory / "timing" / f"rank-{rank}" / "record.json" for rank in self.roles}
        if self.options["host_timing"] or self.options.get("attention_sites"):
            kind = "host_samples" if self.options["host_timing"] else "attention_samples"
            return {kind: {rank: str(path.relative_to(self.output)) for rank, path in records.items()}}

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
        loads = [
            {"rank": record["rank"], **event} for record in measurements for event in record["specialization_loads"]
        ]
        path = self.directory / "latency.json"
        write_json(
            path,
            {"clock": "monotonic_raw_ns", "requests": list(requests.values()), "specialization_loads": loads},
            replace=False,
        )
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
                    windows = NativeProfiles.windows(
                        self.output, plan["observed_roles"], self.local_shards, self.instance_count
                    )
                window = windows[round_index]
                barrier.wait()
                phase = f"capture/{window.observed[0]}" if window.replay else "capture"
                self.generate(engine, phase, capture, clock, barrier)
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
        # A paced workload primes compilation before the uniform warmup and the measured pass.
        phases = ("prime", "warmup") if any(len(arrivals) > 1 for arrivals in capture.groups) else ("warmup",)
        preparation = {phase: EngineSubmissions(self.workload, mode, self.submission_groups) for phase in phases}
        passes = {**preparation, "source": capture}
        count = len(self.workload["requests"])

        # PD bootstrap rooms must stay distinct across the drain and logical cache reset.
        for ordinal, (phase, submissions) in enumerate(passes.items()):
            submissions.prepare(
                RequestSubmissions.observed if phase == "source" else phase, bootstrap, 1 + ordinal * count
            )
        session = {
            "output": str(self.output), "instance_directory": str(self.directory), "mode": mode,
            "rendezvous": {"host": rendezvous["MASTER_ADDR"], "port": int(rendezvous["MASTER_PORT"])},
            "rank_offset": self.rank_offset, "native_participants": self.roles,
            "pp_size": self.pp, "ep_size": self.ep,
            **self.options,
            "instance_count": self.instance_count, "worker_count": self.worker_count,
            **{f"{phase}_requests": [RequestSubmissions.native_id(submissions.prefix, index) for index in range(count)]
               for phase, submissions in passes.items()},
            "representative_warmup": True,
            "protocol_requests": capture.protocol_requests,
            "content": self.workload,
            "request_groups": {phase: submissions.request_groups() for phase, submissions in passes.items()},
            "serving_input": {"configuration": self.configuration, **self.workload},
        }
        write_json(description, session)
        os.environ.update(rendezvous)
        os.environ["PROLOGUE_GRAPH_SESSION"] = str(description)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(self.output / "runtime")
        os.environ["SGLANG_PLUGINS"] = "prologue"
        if self.ep > 1:
            os.environ.update(device_runtime().expert_environment)

        if self.local_shards:
            from prologue.integrations.frameworks.sglang.observation.staging import staged_engine

            Engine = staged_engine()
        else:
            Engine = resolve(symbols().engine)

        options = self.instance.configuration.runner_options(self.model)
        options.update(
            base_gpu_id=0 if self.local_shards else self.rank_offset, nccl_port=int(rendezvous["MASTER_PORT"])
        )
        if mode != "null":
            options.update(
                disaggregation_mode=mode, disaggregation_transfer_backend=platform_domain().pd_transfer,
                disaggregation_bootstrap_port=bootstrap["port"],
                disaggregation_decode_enable_radix_cache=mode == "decode" and self.configuration["prefix_cache"],
            )

        # Native argument normalization can change EP size; reject that change before loading weights or
        # launching workers.
        resolve(symbols().load_plugins)()
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
            warmed_ns = time.monotonic_ns()

            barrier.wait()
            metrics = {}
            if self.options["latency_only"]:
                self.generate(engine, "capture", capture, clock, barrier)
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


class ArrivalClock:
    def __init__(self, context):
        self.origin = context.Value("q", 0)

    def __call__(self):
        # Worker receipts and profiler host anchors use the same machine's unadjusted monotonic clock.
        self.origin.value = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class NativeRendezvous:
    host = "127.0.0.1"

    @classmethod
    @contextmanager
    def loopback_listener(cls):
        # Port zero excludes other listeners but can still select a host NAT destination.
        for _ in range(4):
            with socket.socket() as listener:
                listener.bind((cls.host, 0))
                listener.listen(1)
                listener.settimeout(0.25)
                try:
                    with socket.create_connection(listener.getsockname(), timeout=0.25) as probe:
                        accepted, peer = listener.accept()
                        with accepted:
                            if peer != probe.getsockname():
                                continue

                            # The client closes first so the native server can rebind the checked port.
                            probe.shutdown(socket.SHUT_WR)
                            accepted.settimeout(0.25)
                            accepted.recv(1)
                except (ConnectionError, TimeoutError):
                    continue

                listener.settimeout(None)
                yield listener
                return
        raise RuntimeError("no reachable owned loopback listener for the native session")

    def __init__(self) -> None:
        from torch.distributed import TCPStore

        # TCPStore takes ownership of the checked socket until every Engine worker exits.
        with self.loopback_listener() as listener:
            self.store = TCPStore(self.host, listener.getsockname()[1], is_master=True,
                                  wait_for_workers=False, master_listen_fd=listener.detach())
        self.environment = {
            "MASTER_ADDR": self.host, "MASTER_PORT": str(self.store.port),
            "TORCHELASTIC_USE_AGENT_STORE": "True", "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE": "env://",
        }


class NativeServingSession:
    def __init__(self, job, output, mode, local_shards, attention_sites):
        self.framework_revision = NativeConfiguration.upstream_revision(route().revision)
        serving = CaptureInput(job)
        self.output = output.resolve()
        resource_trace = mode == "resource-trace"
        service_reference = mode == "service-reference"
        host_timing = mode == "host-timing"
        latency_only = mode == "latency-only" or host_timing or bool(attention_sites)
        options = {
            "observe_all_workers": serving.observe_all_workers or service_reference, "local_shards": local_shards,
            "resource_trace": resource_trace or service_reference, "service_reference": service_reference,
            "latency_only": latency_only, "host_timing": host_timing,
        }
        if attention_sites:
            options["attention_sites"] = attention_sites
        self.instances = [NativeServingInstance(serving, instance, self.output, index, options)
                          for index, instance in enumerate(serving.instances)]
        if latency_only and resource_trace:
            raise ValueError("light observation cannot include resource tracing")
        if latency_only and not host_timing and not attention_sites and len(serving.submission_groups) != 1:
            raise ValueError("latency-only measurement requires one submission group")

    @staticmethod
    def execute(instance, barrier, clock, bootstrap, rendezvous, connection):
        # Each Engine owns descendants; cancellation must also stop that instance's native workers.
        try:
            os.setsid()
            connection.send((True, instance.run(barrier, clock, bootstrap, rendezvous)))
        except BaseException:

            # Keep the traceback available before EOF lets the owner cancel this process group.
            connection.send((False, traceback.format_exc()))
            raise
        finally:
            connection.close()

    def run(self) -> None:
        self.output.mkdir(parents=True)
        (self.output / "instances").mkdir()
        (self.output / "runtime").mkdir()
        context = multiprocessing.get_context("spawn")
        clock = ArrivalClock(context)
        barrier = context.Barrier(len(self.instances), action=clock)
        processes, pending, results, rendezvous = [], {}, {}, []
        try:
            rendezvous.extend(NativeRendezvous() for _ in self.instances)

            # Select the native HTTP bootstrap only after all store ports are already occupied.
            bootstrap = None
            if len(self.instances) > 1:
                with NativeRendezvous.loopback_listener() as reservation:
                    host, port = reservation.getsockname()
                    bootstrap = {"host": host, "port": port}

            for index, instance in enumerate(self.instances):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(
                    target=self.execute,
                    args=(instance, barrier, clock, bootstrap, rendezvous[index].environment, writer),
                )
                try:
                    process.start()
                except BaseException:
                    reader.close()
                    raise
                finally:
                    writer.close()
                processes.append(process)
                pending[reader] = index

            while pending:
                for connection in wait(pending):
                    index = pending[connection]
                    try:
                        success, result = connection.recv()
                    except EOFError as error:
                        raise RuntimeError(
                            f"native capture instance {index} exited without reporting a result"
                        ) from error
                    if not success:
                        raise RuntimeError(f"native capture instance {index} failed:\n{result}")
                    results[index] = result
                    del pending[connection]
                    connection.close()
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"native capture worker exited with {process.exitcode}")
            write_json(
                self.output / "capture.json",
                {
                    "framework_revision": self.framework_revision,
                    "instances": [results[index] for index in range(len(results))],
                },
                indent=2,
                replace=False,
            )
        finally:
            for connection in pending:
                connection.close()
            for process in processes:
                if process.exitcode != 0:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        if process.is_alive():
                            process.kill()
                process.join()
            rendezvous.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="run one normalized native observation job")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--local-shards", action="store_true", help="execute logical shards on the single mounted device"
    )
    observation = capture.add_mutually_exclusive_group()
    for mode, help_text in (
        ("resource-trace", "observe resources and input reads for independent comparison"),
        ("service-reference", "observe structure, resources and same-run CPU activity"),
        ("latency-only", "record scheduler ingress and token publication only"),
        ("host-timing", "sample native call cuts and CPU-only queue activity"),
        ("local-copy", "time isolated native tensor copies on one device"),
        ("local-attention", "measure Decode on a native ModelRunner and KV pool"),
        ("local-prebuilt", "observe the receiver assembling a native PD batch"),
        ("local-embedding", "observe a vocabulary lookup across native TP shards"),
        ("local-worker", "read the initialized scheduler, model and pools"),
        ("local-host", "measure local scheduler work with the independent CPU probe"),
        ("local-host-pair", "capture and independently time one local worker cut"),
        ("attention", "measure selected native Decode attention calls"),
    ):
        observation.add_argument("--" + mode, dest="mode", action="store_const", const=mode, help=help_text)
    interpret = commands.add_parser("interpret", help="interpret an SDK-exported worker recording")
    interpret.add_argument("--trace", type=Path, required=True)
    interpret.add_argument(
        "--evidence", action="store_true", help="retain measured boundary clocks for source auditing"
    )
    export = commands.add_parser("export", help="export one worker through an isolated SDK workspace")
    cuts = commands.add_parser("sampling-cuts", help="bind source device work to portable native sampling cuts")
    for option in ("work", "identities", "runner"):
        cuts.add_argument("--" + option, type=Path, required=True)
    for command in (interpret, export, cuts):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--rank", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "capture":
        run_capture(arguments)
    elif arguments.command == "sampling-cuts":
        from prologue.integrations.frameworks.sglang.observation.layout import publish_cuts

        publish_cuts(arguments.input, arguments.work, arguments.rank, arguments.identities, arguments.runner,
                     arguments.output)
    elif arguments.command == "export":
        device_runtime().export.run(arguments.input, arguments.rank, arguments.output)
    else:
        from prologue.integrations.frameworks.sglang.observation.layout import serving_facts

        facts = serving_facts(arguments.input, arguments.rank, arguments.trace, evidence=arguments.evidence)
        write_json(arguments.output, facts, replace=False)


def run_capture(args):
    job = json.loads(args.input.read_bytes())
    route().require_job(job)
    revision = route().revision
    mode = args.mode
    if mode == "local-host-pair" or (mode == "local-worker" and "calibration" in job):
        from prologue.integrations.frameworks.sglang.observation.scheduler_samples import WorkerProbes

        probes = WorkerProbes(revision)
        if mode == "local-host-pair":
            probes.pair(args.input, args.output)
        else:
            probes.calibration(job, args.output)
    elif mode == "local-copy":
        device_runtime().copy_sampler().sample(job, args.output, revision)
    elif mode == "local-attention":
        from prologue.integrations.frameworks.sglang.observation.model_samples import AttentionRunner

        AttentionRunner(job, args.output).run(revision)
    elif mode in {"local-embedding", "local-prebuilt", "local-worker", "local-host"}:

        # Install the queue sink before importing native classes that can initialize the runtime.
        runtime = args.output / "runtime"
        runtime.mkdir(parents=True)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(runtime)
        if mode == "local-embedding":
            from prologue.integrations.frameworks.sglang.observation.model_samples import EmbeddingRunner

            EmbeddingRunner(job, args.output).run(revision)
        else:
            from prologue.integrations.frameworks.sglang.observation.scheduler_samples import (
                PrebuiltRunner,
                SchedulerRunner,
            )

            if mode == "local-prebuilt":
                PrebuiltRunner(job, args.output).run(revision)
            else:
                SchedulerRunner(job, args.output).run(revision, host_timing=mode == "local-host")
    else:
        attention = job.pop("attention_sites") if mode == "attention" else None
        NativeServingSession(job, args.output, mode, args.local_shards, attention).run()
        if attention:

            # Export only after the workers and their observation windows have closed.
            device_runtime().scoped_tasks.publish(args.output)
