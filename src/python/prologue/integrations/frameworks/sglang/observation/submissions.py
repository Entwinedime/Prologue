"""Deliver declared arrivals without waiting for earlier requests to finish."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from prologue.integrations.frameworks.sglang.observation.window import ObservationWindow
from prologue.integrations.frameworks.sglang.shared.batches import RequestSubmissions


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

        rows = []
        async for row in responses:
            if row["meta_info"]["finish_reason"] is not None:
                rows.append(row)
        return rows

    def check_responses(self, prefix: str, rows: list[dict], indices: list[int]) -> None:
        expected = {
            self.ingress.native_id(prefix, index): 1
            if self.mode == "prefill"
            else self.ingress.requests[index]["output_tokens"]
            for index in indices
        }
        tokens = {
            self.ingress.native_id(prefix, index): self.ingress.content.output_token(self.ingress.requests[index]["id"])
            for index in indices
        }
        completed = set()
        for row in rows:
            metadata = row["meta_info"]
            identity = metadata["id"]
            reason = metadata["finish_reason"]
            if reason is None or reason["type"] != "length" or metadata["completion_tokens"] != expected[identity]:
                raise RuntimeError(f"native request {identity} did not finish its fixed output: {reason}")
            if any(token != tokens[identity] for token in row["output_ids"]):
                raise RuntimeError(f"native request {identity} did not preserve its declared synthetic output")
            if identity in completed:
                raise RuntimeError(f"native request completed twice: {identity}")
            completed.add(identity)
        if completed != expected.keys():
            raise RuntimeError(f"native responses omitted requests: {sorted(expected.keys() - completed)}")
