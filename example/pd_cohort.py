"""Observe same-batch native PD admission using the stock pause and resume interface."""

import argparse
import asyncio
import json
import multiprocessing
from itertools import groupby
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns, monotonic

from prologue.integrations.frameworks.sglang.configuration import Disaggregation, Model
from prologue.integrations.frameworks.sglang.observation.disaggregation import DisaggregatedSession, ServingRole


class CohortRole(ServingRole):
    def prepare(self, **details):
        return super().prepare(**details, stream_handoff=self.role == "decode")

    async def dispatch(self, engine, phase):
        if self.role == "decode":
            for gate in self.pair.paused.values():
                gate.clear()
        return await super().dispatch(engine, phase)

    def witnessed(self, path, kind, members, since):
        if not path.is_file():
            return False
        with path.open(encoding="utf-8") as events:
            observed = {event.get("request") for line in events if line.endswith("\n")
                        if (event := json.loads(line))["kind"] == kind and event["at_ns"] >= since}
        return members <= observed

    async def waiting(self, phase, kind, members, seconds, *, since=0):
        window = self.output / f"window-{phase.removeprefix('source-')}"
        files = [window / f"rank-{rank}" / "handoff-live.jsonl" for rank in range(self.configuration.tp)]
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            if all(self.witnessed(path, kind, members, since) for path in files):
                return
            await asyncio.sleep(0.01)
        raise TimeoutError(f"native Decode did not report {kind} for {members}")

    async def generate(self, engine, phase, requests, rooms):
        if phase == "warmup" or len(requests) == 1:
            return await super().generate(engine, phase, requests, rooms)

        gate = self.pair.paused[requests[0][0]]
        if self.role == "prefill":
            if not await asyncio.to_thread(gate.wait, 30):
                raise TimeoutError("Decode did not pause before the cohort Prefill")
            return await super().generate(engine, phase, requests, rooms)

        from sglang.srt.managers.io_struct import ContinueGenerationReqInput, PauseGenerationReqInput

        manager = engine.tokenizer_manager
        members = {request["id"] for _, request in requests}
        pending = asyncio.create_task(super().generate(engine, phase, requests, rooms))
        await self.waiting(phase, "reserve", members, 12)
        closed = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        await manager.pause_generation(PauseGenerationReqInput(mode="in_place"))
        try:
            await self.waiting(phase, "paused", {None}, 12, since=closed)
            gate.set()
            await self.waiting(phase, "commit", members, 35)
        finally:
            opened = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            await manager.continue_generation(ContinueGenerationReqInput(torch_empty_cache=False))
            await self.waiting(phase, "resumed", {None}, 12, since=opened)
            with (self.output / "admission-release.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"phase": phase, "begin_ns": opened,
                                         "end_ns": clock_gettime_ns(CLOCK_MONOTONIC_RAW)}) + "\n")
        return await pending


class CohortSession(DisaggregatedSession):
    role_class = CohortRole

    def __init__(self, model, configuration, workload, output, placement):
        super().__init__(model, configuration, workload, output, "detailed", placement)
        context = multiprocessing.get_context("spawn")
        self.paused = {}
        arrivals = groupby(enumerate(workload["requests"]), key=lambda item: item[1]["arrival_ns"])
        for _, tied in arrivals:
            cohort = list(tied)
            if len(cohort) > 1:
                self.paused[cohort[0][0]] = context.Event()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--placement", choices=("single_device", "full"), required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    configuration, = plan["configurations"]
    workload, = plan["workloads"]
    CohortSession(Model(**plan["model"]), Disaggregation(**configuration), workload, args.output, args.placement).run()


if __name__ == "__main__":
    main()
