"""Translate simulated arrivals at the native tokenizer receive boundary."""

import sys

import zmq

from prologue.common.workload.routing import RoutedContent
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.cache_keys import RoutedCacheKeys
from prologue.integrations.frameworks.sglang.shared.ingress import RequestSubmissions
from prologue.integrations.frameworks.sglang.shared.pipeline import PipelineABI

DisaggregationMode = resolve(symbols().disaggregation_mode)
BatchTokenizedGenerateReqInput = resolve(symbols().batch_tokenized_generate_req_input)


class RequestInput(RoutedCacheKeys):
    def __init__(
        self, peer, vocab_size: int, workload: dict, routes, mode: str, expert_layers: set[int]
    ) -> None:
        self.peer = peer
        self.vocab_size = vocab_size
        self.mode = DisaggregationMode(mode)
        self.arrivals = None
        self.waiting = False
        self.poll_deadline_ns = None
        self.context = None
        self.in_flow = False
        self.submissions = RequestSubmissions(workload)
        self.submissions.content.check_vocabulary(vocab_size)
        self.routing = (
            RoutedContent(self.submissions.content, routes["assignments"], expert_layers)
            if routes is not None
            else None
        )
        if self.routing is not None:
            self.install_cache_keys()

    def recv_pyobj(self, flags):
        if self.arrivals is None:
            advance = self.peer.call("input", waiting=self.waiting, poll_deadline_ns=self.poll_deadline_ns,
                                     context=self.context, in_flow=self.in_flow)
            if advance["kind"] != "advance":
                raise ValueError("Native control expected an advance boundary")
            # One socket receive may read several dispatches. Each keeps its own tokenizer envelope.
            self.arrivals = iter(self.submissions.cohorts(advance["arrivals"]))

        cohort = next(self.arrivals, None)

        # End this native receive poll; the next poll waits for core progress.
        if cohort is None:
            self.arrivals = None
            raise zmq.Again()

        _, indices = cohort
        return BatchTokenizedGenerateReqInput(batch=[self.tokenize(index) for index in indices])

    def tokenize(self, index):
        request = self.submissions.requests[index]
        return self.submissions.tokenized(
            self.submissions.native_id(self.submissions.observed, index),
            self.submissions.content.prompt(request["id"]),
            self.submissions.sampling(index),
            self.vocab_size,
            self.mode,
            extra_key=request["id"] if self.routing is not None else None,
            room=index + 1 if self.mode != DisaggregationMode.NULL else None,
        )

    def request_id(self, identity):
        return self.submissions.requests[self.submissions.index(identity)]["id"]

    def output_token(self, identity: str) -> int:
        return self.submissions.content.output_token(self.request_id(identity))


class NativeRequestReceiver:
    def __init__(self, receiver, scheduler) -> None:
        self.receiver = receiver
        self.scheduler = scheduler
        self.round_started = False
        self.model_ran = False
        self.idle_reported = False
        self.poll_deadline_ns = None

    def poll(self, timeout):
        # SGLang supplies the socket timeout. Simulated time advances at its next receive boundary.
        if self.scheduler.calls.started and timeout >= 0:
            self.poll_deadline_ns = self.scheduler.peer.call("clock") + timeout * 1_000_000

    def recv_requests(self):
        scheduler = self.scheduler

        # The native loop owns this cursor. A PP round includes every slot, not each individual receive.
        first_slot = scheduler.ps.pp_size == 1 or sys._getframe(1).f_locals["mb_id"] == 0
        inbox = scheduler.request_input
        inbox.waiting = False
        inbox.poll_deadline_ns = None
        if first_slot:

            # Native on_idle can precede round-tail work; park only after that work reaches the next receive.
            idle = self.idle_reported and scheduler.is_fully_idle() and not PipelineABI.pending_model_sends(scheduler)
            self.idle_reported = False
            if idle:
                if scheduler.ps.pp_rank == 0 and scheduler.ps.tp_rank == 0:
                    if scheduler.calls.started:

                        # Between arrivals the native loop still polls. The read owns its return and wake conditions.
                        inbox.waiting = True
                        inbox.poll_deadline_ns = self.poll_deadline_ns
                    else:
                        scheduler.peer.call("idle", waiting=True)
                else:
                    scheduler.peer.send({"kind": "idle", "waiting": False})
            self.poll_deadline_ns = None

            # Pending transfers can bypass on_idle; completed rounds must still have a safe external wait.
            inbox.waiting |= (
                self.round_started and not self.model_ran
                and not scheduler.waiting_queue and scheduler.chunked_req is None
                and scheduler.pd is not None and scheduler.pd.waiting()
            )
            self.round_started = True
            self.model_ran = False
        inbox.context = scheduler.calls.scheduler_work()["context"]
        inbox.in_flow = scheduler.calls.started
        return self.receiver.recv_requests()
