"""Capture the two native serving roles against one workload clock."""

import asyncio
import json
import multiprocessing
import os
import signal
import socket
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from itertools import groupby
from multiprocessing.connection import wait
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.integrations.frameworks.sglang.configuration import native_options
from prologue.integrations.frameworks.sglang.observation.parallel import RankPlan
from prologue.integrations.frameworks.sglang.observation.session import Session


class ArrivalClock:
    def __init__(self, context):
        self.origin = context.Value("q", 0)

    def start(self):
        self.origin.value = clock_gettime_ns(CLOCK_MONOTONIC_RAW)


class ServingRole(Session):
    def __init__(self, pair, role, clock, barrier, bootstrap, transport_store, nccl, grpc, offset):
        super().__init__(pair.model, pair.configuration.instances[role], pair.workload, pair.output / role,
                         pair.mode, pair.placement)
        self.pair = pair
        self.role = role
        self.ranked = True
        self.clock, self.barrier = clock, barrier
        self.bootstrap = bootstrap
        self.transport_store = transport_store
        self.nccl, self.grpc = nccl, grpc
        self.offset = offset
        self.submissions = 0

    async def generate(self, engine, phase, requests, rooms):
        submitted = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        stream = await engine.async_generate(
            rid=[f"{phase}/{index}" for index, _ in requests],
            input_ids=[[1] * request["prompt_tokens"] for _, request in requests],
            sampling_params=[{"max_new_tokens": request["output_tokens"], "ignore_eos": True, "temperature": 0}
                             for _, request in requests],
            bootstrap_host="127.0.0.1", bootstrap_port=self.bootstrap,
            bootstrap_room=[rooms + index for index, _ in requests],
            stream=True,
        )
        finished = {}
        async for row in stream:
            metadata = row["meta_info"]
            if metadata["finish_reason"] is not None:
                finished[metadata["id"]] = metadata["completion_tokens"]

        delivered = []
        for index, request in requests:
            expected = request["output_tokens"] if self.role == "decode" else 1
            if finished.get(f"{phase}/{index}") != expected:
                raise ValueError(f"native {self.role} did not complete request {request['id']}")
            delivered.append({"request": request["id"], "submitted_raw_ns": submitted,
                              "offset_ns": request["arrival_ns"]})
        return delivered

    async def dispatch(self, engine, phase):
        self.barrier.wait()
        origin = self.clock.origin.value
        requests = self.workload["requests"]
        rooms = self.submissions * len(requests) + 1
        self.submissions += 1
        pending = []
        for offset, group in groupby(enumerate(requests), key=lambda item: item[1]["arrival_ns"]):
            delay = origin + offset - clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            if delay > 0:
                await asyncio.sleep(delay / 1_000_000_000)
            pending.append(asyncio.create_task(self.generate(engine, phase, list(group), rooms)))
        delivered = [row for group in await asyncio.gather(*pending) for row in group]
        return {"origin_raw_ns": origin, "requests": delivered}

    def check_publications(self, directory):
        if self.role == "decode":
            super().check_publications(directory)

    def record_workers(self, engine):
        self.barrier.wait()
        plans = {role: RankPlan(self.pair.output / role, configuration, self.placement)
                 for role, configuration in self.pair.configuration.instances.items()}
        plan = plans[self.role]
        if self.placement == "full":
            windows = plan.windows
        else:
            windows = [ranks if role == self.role else [] for role, group in plans.items() for ranks in group.windows]
        recorded = [self.record_ranks(engine, index, ranks, plan.publisher) for index, ranks in enumerate(windows)]
        return {"workers": plan.workers, "windows": recorded}

    def engine_options(self):
        options = native_options(self.model, self.configuration)
        options.update(self.pair.configuration.native_policy(self.role),
                       disaggregation_bootstrap_port=self.bootstrap, base_gpu_id=self.offset, nccl_port=self.nccl)
        return options

    def run(self):
        os.environ.update(ASCEND_MF_STORE_URL=f"tcp://127.0.0.1:{self.transport_store}",
                          ASCEND_MF_TRANSFER_PROTOCOL="sdma", SGLANG_GRPC_PORT=str(self.grpc))
        Engine = self.prepare(disaggregation=self.role, device_lock=str(self.pair.output / ".device.lock"))
        started = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        engine = Engine(**self.engine_options())
        try:
            initialized = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            engine.loop.run_until_complete(self.dispatch(engine, "warmup"))
            self.drain(engine)
            warmed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            recorded = self.record_workers(engine)
            self.barrier.wait()
        finally:
            engine.shutdown()
            if engine.tokenizer_manager.bootstrap_server is not None:
                engine.tokenizer_manager.bootstrap_server.close()
        self.save(started, initialized, warmed, recorded)

class DisaggregatedSession:
    role_class = ServingRole

    def __init__(self, model, configuration, workload, output, mode, placement):
        if mode != "detailed":
            raise ValueError("disaggregated capture currently requires detailed observation")
        configuration.route_table(workload)
        self.model, self.configuration, self.workload = model, configuration, workload
        self.output, self.mode, self.placement = output.resolve(), mode, placement

    @staticmethod
    def allocate_ports(count):
        with ExitStack() as stack:
            ports = []
            for _ in range(count):
                listener = stack.enter_context(socket.socket())
                listener.bind(("127.0.0.1", 0))
                ports.append(listener.getsockname()[1])
            return ports

    @staticmethod
    def execute(role, pipe):
        with pipe:
            os.setsid()
            try:
                role.run()
                pipe.send(None)
            except BaseException:
                pipe.send(traceback.format_exc())
                raise

    def run(self):
        self.output.mkdir()
        context = multiprocessing.get_context("spawn")
        clock = ArrivalClock(context)
        barrier = context.Barrier(len(self.configuration.roles), action=clock.start)
        bootstrap, transport_store, *ports = self.allocate_ports(2 + 2 * len(self.configuration.roles))
        processes, pending = [], {}
        offset = 0
        try:
            for role in self.configuration.roles:
                read, write = context.Pipe(duplex=False)
                instance = self.role_class(self, role, clock, barrier, bootstrap, transport_store,
                                           ports.pop(), ports.pop(), offset if self.placement == "full" else 0)
                process = context.Process(target=self.execute, args=(instance, write), name=role)
                with write:
                    process.start()
                pending[read] = role
                processes.append(process)
                offset += instance.configuration.worker_count
            while pending:
                for pipe in wait(pending):
                    role = pending.pop(pipe)
                    with pipe:
                        error = pipe.recv()
                    if error is not None:
                        raise RuntimeError(f"native {role} capture failed:\n{error}")
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"native {process.name} capture exited with status {process.exitcode}")
        finally:
            for pipe in pending:
                pipe.close()
            for process in processes:
                if process.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        process.terminate()
                    process.join()

        document = {"framework_revision": os.environ["PROLOGUE_SGLANG_REVISION"], "model": asdict(self.model),
                    "configuration": self.configuration.document(), "workload": self.workload,
                    "placement": self.placement, "mode": self.mode, "roles": list(self.configuration.roles)}
        (self.output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
