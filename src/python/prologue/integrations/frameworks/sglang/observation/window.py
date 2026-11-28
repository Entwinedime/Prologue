"""Open the native observation window and submit its paced request groups."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prologue.common.io import write_json
from prologue.common.observation import ProfileWindow
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.batches import RequestSubmissions


class ObservationWindow:
    def __init__(self, directory: Path, description: dict[str, Any]) -> None:
        self.directory = directory
        self.participants = description["native_participants"]
        self.requests = {phase: set(description[f"{phase}_requests"]) for phase in description["request_groups"]}
        self.pending = {phase: set(requests) for phase, requests in self.requests.items()}
        self.groups = {
            phase: [set(group) for group in groups[:-1]] for phase, groups in description["request_groups"].items()
        }
        self.next_group = dict.fromkeys(self.groups, 0)

        self.phase: str | None = None
        self.drained = False
        self.prepared_phase: str | None = None
        self.receipts: list[dict[str, Any]] = []
        self.receive_rounds: list[dict[str, Any]] = []
        self.ingress: list[dict[str, Any]] = []
        self.input_attempts: list[dict[str, Any]] = []

    def begin(self, identity: str) -> None:
        for phase, requests in self.requests.items():
            if identity in requests:
                self.pending[phase].discard(identity)
                self.phase = phase
                self.drained = False

                if phase == "source":

                    # This hook runs at native request dispatch, after receive and rank broadcast.
                    self.receipts.append(
                        {"request": identity, "received_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}
                    )

                return

    def receive(self, requests: list[str], time_ns: int, *, ingress: bool, attempt=None) -> None:
        observed = [request for request in requests if request in self.requests["source"]]
        if attempt is not None and (observed or (self.phase == "source" and not self.drained)):

            # Keep the first source pull and in-window empty reads, not idle polls before or after the window.
            self.input_attempts.append({**attempt, "requests": observed, "received_raw_ns": time_ns})
        if observed:

            # One receive round may admit several requests; it does not prescribe their execution batches.
            records = self.ingress if ingress else self.receive_rounds
            records.append({"requests": observed, "received_raw_ns": time_ns})

    @staticmethod
    def marker(directory: Path, rank: int) -> Path:
        return directory / "runtime" / f"window-rank-{rank}.json"

    def inputs_complete(self) -> bool:
        return self.phase is not None and not self.pending[self.phase]

    def group_idle(self, rank: int) -> None:
        phase = self.phase
        if phase is None:
            return
        group = self.next_group[phase]
        if group == len(self.groups[phase]) or self.groups[phase][group] & self.pending[phase]:
            return

        # A terminal response can precede native removal from running batches; preserve cache while waiting for idle.
        write_json(self.marker(self.directory, rank), {"phase": phase, "state": f"group/{group}"})
        self.next_group[phase] += 1

    def drain(self, rank: int) -> None:
        write_json(self.marker(self.directory, rank), {"phase": self.phase, "state": "drained"})
        self.drained = True

    def complete(self, rank: int, ingress: bool) -> None:
        if ingress and not self.ready(
            self.directory, (peer for peer in self.participants if peer != rank), self.prepared_phase, "complete"
        ):
            return

        # Publish at the native whole-loop idle boundary, after reset and observer preparation.
        write_json(self.marker(self.directory, rank), {"phase": self.prepared_phase, "state": "complete"})
        self.prepared_phase = None
        self.phase = None
        self.drained = False

    @classmethod
    def ready(cls, directory: Path, ranks: Iterable[int], phase: str, state: str) -> bool:
        markers = [cls.marker(directory, rank) for rank in ranks]
        return all(
            path.is_file() and json.loads(path.read_text()) == {"phase": phase, "state": state} for path in markers
        )

    @classmethod
    def wait(cls, directory: Path, ranks: Iterable[int], phase: str, state: str) -> None:
        ranks = tuple(ranks)

        while not cls.ready(directory, ranks, phase, state):
            time.sleep(0.05)


class NativeProfiles:
    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.windows_by_round = None
        self.round = 0
        from prologue.integrations.runtimes.pytorch.capture import CpuProtocol

        capture = observation.capture
        self.protocol = CpuProtocol(
            observation.description["native_participants"], observation.calls.communicating,
            capture.current, capture.runtime.service_scope,
            transport_timing=observation.description["service_reference"],
        )

    @staticmethod
    def windows(directory, observed, single_device, instances):
        if not single_device:
            return [ProfileWindow(tuple(observed))]
        if instances > 1:

            # Both PD pools execute every request. Exactly one role across both pools owns the device profiler.
            plans = [directory / "instances" / str(index) / "role-plan.json" for index in range(instances)]
            while not all(path.is_file() for path in plans):
                time.sleep(0.05)
            observed = [rank for path in plans for rank in json.loads(path.read_bytes())["observed_roles"]]
        return [ProfileWindow((rank,), replay=True) for rank in observed]

    @staticmethod
    def wait(directory: Path, ranks, participants, protocol_directory) -> None:
        published = [directory / "native" / f"rank-{rank}" for rank in ranks]
        protocols = [protocol_directory / f"rank-{rank}.json" for rank in participants]

        # A new observer can take the device only after the previous profiler closed and every participant published.
        while not all(path.is_dir() for path in published) or not all(path.is_file() for path in protocols):
            time.sleep(0.05)

    def request(self, original: Any, manager: Any, request: Any) -> Any:
        ProfileReqType = resolve(symbols().profile_req_type)

        owner = self.observation
        if request.req_type == ProfileReqType.START_PROFILE:

            # All instances have drained warmup and published their real pool layouts before this request.
            if self.windows_by_round is None:
                self.windows_by_round = self.windows(
                    owner.directory,
                    owner.plan_roles(),
                    owner.description["local_shards"],
                    owner.description["instance_count"],
                )
            else:

                # The preceding request drained and flushed its KV. This replay owns fresh ingress and group receipts.
                owner.window = ObservationWindow(owner.directory, owner.description)

            window = self.windows_by_round[self.round]
            owner.selected = owner.parallel["rank"] in window.observed
            self.protocol_root = window.protocol_directory(owner.directory)
            if window.replay:

                # A later replay can reuse a registration with a different request slot. Keep this window's peer
                # facts together.
                owner.kv_pools = self.protocol_root / "kv-pools"

        if owner.selected:
            result = original(manager, request)
        else:
            if request.req_type == ProfileReqType.STOP_PROFILE:
                import torch.distributed as distributed

                # Non-exporting participants still join the native profiler's stop consensus.
                distributed.barrier(manager.dp_tp_cpu_group)

            result = resolve(symbols().profile_req_output)(success=True, message="Worker has no assigned observation")

        if request.req_type == ProfileReqType.START_PROFILE and result.success:
            self.protocol.begin()
            owner.window.prepared_phase = "profiler"
        elif request.req_type == ProfileReqType.STOP_PROFILE and result.success:

            # Protocol facts cover every participant, including workers represented by another device profile.
            write_json(
                self.protocol_root / f"rank-{owner.parallel['rank']}.json", self.protocol.document(), replace=False
            )
            self.round += 1

        return result

    def start(self, original: Any, manager: Any, output_dir: str, *args: Any, **kwargs: Any) -> Any:
        rank = self.observation.parallel["rank"]
        self.destination = Path(output_dir) / f"rank-{rank}"
        return original(manager, str(self.destination.with_name(f".rank-{rank}")), *args, **kwargs)

    def stop(self, original: Any, manager: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(manager, *args, **kwargs)
        if result.success:
            manager.torch_profiler_output_dir.rename(self.destination)
            if self.observation.capture.closed:
                self.observation.capture.runtime.release()
        return result

    def install(self) -> None:
        manager = symbols().scheduler_profiler_manager + "."
        HookTable.group({
            manager + "_profile": self.request,
            manager + "_init_profile": self.start,
            manager + "_stop_profile": self.stop,
        }).install()


class EngineSubmissions:

    def __init__(self, source: dict[str, Any], mode: str, groups: list[list[str]] | None = None) -> None:
        self.mode = mode
        self.ingress = RequestSubmissions(source)
        indices = {request["id"]: index for index, request in enumerate(self.ingress.requests)}
        self.groups = [self.ingress.cohorts(indices[identity] for identity in group)
                       for group in (groups if groups is not None else [list(indices)])]
        self.delivery: list[dict[str, Any]] = []

    def prepare(self, prefix: str, bootstrap: dict | None, first_room: int) -> None:
        self.prefix = prefix
        self.protocol_requests = {}

        # Construct prompts before observation; each group preserves its own declared arrival offsets.
        requests = self.ingress.requests
        prompts = [self.ingress.content.prompt(request["id"]) for request in requests]
        self.submissions = []
        for group, arrivals in enumerate(self.groups):
            self.submissions.append([])
            for arrival, indices in arrivals:
                payload = self.ingress.payload(prefix, indices, prompts)
                if self.mode != "null":
                    payload.update(
                        bootstrap_host=[bootstrap["host"]] * len(indices),
                        bootstrap_port=[bootstrap["port"]] * len(indices),
                        bootstrap_room=[first_room + index for index in indices],
                    )

                    # Workers can receive remote reservations before dispatching their own copy of the request.
                    self.protocol_requests.update(
                        (str(room), requests[index]["id"])
                        for room, index in zip(payload["bootstrap_room"], indices, strict=True)
                    )

                self.submissions[-1].append((payload, {"group": group, "arrival_ns": arrival, "requests": indices}))

    def request_groups(self) -> list[list[str]]:
        return [[identity for payload, _ in group for identity in payload["rid"]] for group in self.submissions]

    async def run(self, endpoint: Any, clock: Any, barrier: Any, directory, ranks) -> None:
        origin_ns = clock.origin.value
        phase = "source" if self.prefix == RequestSubmissions.observed else self.prefix

        # A second observer replays the same declared arrivals and gets its own measured delivery times.
        groups = [[(payload, dict(record)) for payload, record in group] for group in self.submissions]
        records = [record for group in groups for _, record in group]
        self.delivery = [{"clock": "monotonic_raw_ns", "origin_ns": origin_ns, "submissions": records}]
        for index, group in enumerate(groups):
            if index:

                # Worker idle, not client completion, closes a group; both PD instances join before advancing time.
                await asyncio.to_thread(ObservationWindow.wait, directory, ranks, phase, f"group/{index - 1}")
                barrier.wait()

            offset_ns = clock.origin.value - origin_ns
            for _, record in group:
                record["scheduled_ns"] = offset_ns + record["arrival_ns"]
            submission_order = asyncio.Lock()
            tasks = [
                asyncio.create_task(self.submit_at(endpoint, payload, record, origin_ns, submission_order))
                for payload, record in group
            ]

            try:
                responses = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            indices = [request for _, record in group for request in record["requests"]]
            self.check_responses(self.prefix, [row for response in responses for row in response], indices)

    async def submit_at(
        self, endpoint: Any, payload: dict, record: dict, started_ns: int, submission_order: asyncio.Lock,
    ) -> list[dict]:
        deadline_ns = started_ns + record["scheduled_ns"]

        # A FIFO lock orders the deadlines and Engine calls even when timer wakeups bunch together.
        async with submission_order:
            while (remaining_ns := deadline_ns - time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)) > 0:
                await asyncio.sleep(remaining_ns / 1e9)
            record["submitted_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started_ns
            responses = await endpoint.async_generate(**payload)

        return [row async for row in responses if row["meta_info"]["finish_reason"] is not None]

    def check_responses(self, prefix: str, rows: list[dict], indices: list[int]) -> None:
        expected = {}
        for index in indices:
            request = self.ingress.requests[index]
            length = 1 if self.mode == "prefill" else request["output_tokens"]
            expected[self.ingress.native_id(prefix, index)] = length, self.ingress.content.output_token(request["id"])

        completed = set()
        for row in rows:
            metadata = row["meta_info"]
            identity = metadata["id"]
            reason = metadata["finish_reason"]

            if reason is None or reason["type"] != "length" or metadata["completion_tokens"] != expected[identity][0]:
                raise RuntimeError(f"native request {identity} did not finish its fixed output: {reason}")
            if any(token != expected[identity][1] for token in row["output_ids"]):
                raise RuntimeError(f"native request {identity} did not preserve its declared synthetic output")
            if identity in completed:
                raise RuntimeError(f"native request completed twice: {identity}")

            completed.add(identity)

        if completed != expected.keys():
            raise RuntimeError(f"native responses omitted requests: {sorted(expected.keys() - completed)}")


class LocalProbe:
    def __init__(self, job, output):
        self.job, self.output = job, output

    @staticmethod
    def runner_args(native, configuration, model):
        args = native.ServerArgs(**configuration.runner_options(model))
        native._set_envs_and_config(args)
        return args

    def publish(self, revision, starts, **evidence):
        publish_capture(self.output, revision, starts, **evidence)


def activate_device(rank, world, staged):
    device = 0 if staged else rank
    runtime = device_runtime()
    runtime.bind_worker(device)

    if runtime.api.device_count() != (1 if staged else world):
        raise ValueError("visible device count differs from the requested placement")
    runtime.api.set_device(device)
    return device


def publish_facts(directory, rank, raw, sdk, name):
    from prologue.integrations.frameworks.sglang.observation.layout import observed_graph

    device_runtime().export.run(directory, rank, sdk)
    facts = observed_graph(directory, rank, raw, sdk / "trace.json", evidence=False).document()
    write_json(directory / name, facts)


def publish_capture(directory, revision, starts, **evidence):
    write_json(directory / "capture.json", {
        "framework_revision": revision, "engine_starts": 0, "model_runner_starts": starts, **evidence,
    })
