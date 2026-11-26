"""Observe request ingress and bind native batch and request state."""

from __future__ import annotations

import importlib
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any

from prologue.common.workload import WorkloadContent
from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols


class BatchTensors:
    @staticmethod
    def bind_location_rows(scope, slots, writes):
        members = scope["work"]["members"]
        owners = {slot: member["id"] for slot, member in zip(slots, members, strict=True)}
        if len(owners) != len(members):
            raise ValueError("admitted members share a request-table slot")

        output, = (port["tensor"] for port in scope["ports"]
                   if port["name"] == "cache_locations" and port["direction"] == "output")

        # The native row selects its request. The template keeps the packing rule, independently of physical slots.
        output["packing"] = {
            "table": "request_to_token",
            "member_count": "members", "row_extent": "sequence_capacity",
            "range": {"begin": "context_tokens", "end": ["add", "context_tokens", "new_tokens"]},
            "slices": [{"request": owners[row["slot"]], "source": row["source"], "target": row["target"]}
                       for row in writes],
        }

    @staticmethod
    def boundaries(batch: Any) -> dict[str, tuple[Any, list[str]]]:

        # Capture and control publish the same batch ABI; native scratch tensors stay inside the call.
        return {
            "request_slots": (batch.req_pool_indices, ["members"]),
            "sequence_lengths": (batch.seq_lens, ["members"]),
            "cache_locations": (batch.out_cache_loc, ["input_tokens"]),
        }

    @staticmethod
    def values(batch: Any) -> dict[str, Any]:
        return {name: tensor for name, (tensor, _) in BatchTensors.boundaries(batch).items()}

    @staticmethod
    def phase(batch: Any) -> str:
        if batch.forward_mode.is_mixed():
            return "mixed"
        return "prefill" if batch.forward_mode.is_extend() or batch.forward_mode.is_prebuilt() else "decode"

    @staticmethod
    def members(batch: Any) -> list[dict[str, Any]]:
        sequences = batch.seq_lens_cpu.tolist()
        if batch.forward_mode.is_extend() or batch.forward_mode.is_prebuilt():
            lengths = batch.extend_lens
        else:
            lengths = [1] * len(batch.reqs)

        decoding = {request.rid for request in batch.decoding_reqs or ()}
        return [
            {
                "id": request.rid, "new_tokens": int(tokens), "context_tokens": int(sequence - tokens),
                "slot": int(request.req_pool_idx),
                "phase": "decode" if batch.forward_mode.is_decode() or request.rid in decoding else "prefill",
            }
            for request, sequence, tokens in zip(batch.reqs, sequences, lengths, strict=True)
        ]

    @staticmethod
    def input_dimensions(tokens: int, members: int) -> dict[str, int]:
        return {"input_tokens": int(tokens), "members": int(members)}

    @staticmethod
    def parameters(batch: Any, tokens: int) -> dict[str, int]:
        dimensions = BatchTensors.input_dimensions(tokens, len(batch.reqs))
        return {
            **BatchTensors.capacity(batch.token_to_kv_pool_allocator, batch.req_to_token_pool),
            **dimensions,
            "model_tokens": dimensions["input_tokens"], "model_members": dimensions["members"],
        }

    @staticmethod
    def capacity(pool, slots):
        mapping = slots.req_to_token
        return {
            "kv_pages": pool.size // pool.page_size, "page_size": pool.page_size,
            "slot_capacity": mapping.shape[0] - 1, "sequence_capacity": mapping.shape[1],
        }

    @staticmethod
    def align_model_tokens(tokens: int, alignment: int) -> int:
        return (tokens + alignment - 1) // alignment * alignment

    @staticmethod
    def model_parameters(batch: Any, tokens: int) -> dict[str, int]:
        model_tokens, model_members = tokens, len(batch.reqs)
        if batch.global_num_tokens is not None:
            get_parallel = resolve(symbols().get_parallel)

            # Native eager MLP-sync pads token rows and, for Decode, the model's batch metadata as well.
            parallel = get_parallel()
            if len(batch.global_num_tokens) != 1 or parallel.attn_cp_size != 1:
                raise ValueError("model padding requires a declared DP/CP layout")
            model_tokens = BatchTensors.align_model_tokens(batch.global_num_tokens[0], parallel.attn_tp_size)

            if batch.forward_mode.is_decode():
                model_members = model_tokens

        return {"model_tokens": model_tokens, "model_members": model_members}

    @staticmethod
    def model_padding_guards(parameters: dict[str, int]) -> list[list[str]]:

        # Padding zero rows and materializing a positive tail run different native work.
        dimensions = (("input_tokens", "model_tokens"), ("members", "model_members"))
        return [["eq" if parameters[actual] == parameters[padded] else "lt", actual, padded]
                for actual, padded in dimensions]

    @staticmethod
    def attention_layout(backend, parameters, members):
        if backend not in platform_domain().strided_attention:
            return {}

        # This backend indexes a dense [batch, longest] table, then takes every page_size-th column.
        longest = max((member["context_tokens"] + member["new_tokens"] for member in members), default=0)
        page = parameters["page_size"]
        strided = page > 1 and longest > 1 and (parameters["model_members"] > 1 or longest > page)
        return {"block_table_strided": int(strided)}


class RoutedCacheKeys:
    def install_cache_keys(self):
        HookTable.around(symbols().radix_key_init, self.cache_key)

    def cache_key(self, original, key, token_ids, extra_key=None, is_bigram=False, limit=None):
        if extra_key is not None:
            token_ids = self.routing.key(extra_key, token_ids if limit is None else token_ids[:limit])
            extra_key = None
        original(key, token_ids, extra_key, is_bigram, limit)


class ClockNamespace:
    def __init__(self, owner, native, module) -> None:
        self.owner, self.native, self.module = owner, native, module

    def __getattr__(self, name):
        consumer = self.owner.consumer.get()
        if (
            consumer is not None
            and consumer.startswith(self.module + ".")
            and name in {"time", "monotonic", "perf_counter"}
        ):
            return partial(self.owner.read, name, consumer)
        return getattr(self.native, name)


class DecisionClocks:
    wait_queue_entry = "SchedulerReqTimeStats.set_wait_queue_entry_time"

    def __init__(self, calls) -> None:
        self.calls = calls
        self.consumer = ContextVar("native_decision_clock", default=None)

    def read(self, name, consumer):
        return self.calls.clock_read(getattr(time, name), {"kind": "clock", "clock": name, "consumer": consumer})

    def invoke(self, consumer, original, *args, **kwargs):
        previous = self.consumer.set(consumer)
        try:
            return original(*args, **kwargs)
        finally:
            self.consumer.reset(previous)

    def install(self):

        # These functions create or consume policy timestamps; metrics and background/watchdog clocks stay native.
        consumers = {
            symbols().scheduler_module: (
                "Scheduler._abort_on_running_timeout", "Scheduler._abort_on_waiting_timeout",
            ),
            symbols().req_time_stats: (
                self.wait_queue_entry, "SchedulerReqTimeStats.set_forward_entry_time",
            ),
            symbols().decode: ("DecodePreallocQueue._ensure_prefill_info",),
            symbols().conn: (
                "CommonKVSender._check_bootstrap_timeout", "CommonKVReceiver._check_waiting_timeout",
            ),
            symbols().mooncake_conn: (
                "MooncakeKVSender.__init__", "MooncakeKVReceiver.send_metadata",
            ),
        }

        for name, methods in consumers.items():
            module = importlib.import_module(name)
            module.time = ClockNamespace(self, module.time, module.__name__)
            for method in methods:
                target = module.__name__ + "." + method
                HookTable.around(target, partial(self.invoke, target))


class BatchOperands:
    queues = ("running", "previous", "waiting", "chunked")

    @classmethod
    def queue_input(cls, query):
        contexts = [parent["context"] for parent in query.get("callers", ()) if parent["owner"] == "selection"]
        contexts.append(query.get("context", {}))
        for context in contexts:
            if all(name in context for name in cls.queues) and not context["chunked"]:
                return {name: context[name] for name in cls.queues}

    @classmethod
    def cut(cls, query):
        action = query.get("action")
        condition = query.get("condition", {})
        members = query.get("members", ())
        owners = {parent["owner"] for parent in query.get("callers", ())}

        if action == "segment":
            owners.add(condition.get("owner"))

        # Nested frees and publications run inside the same result call. Its batch supplies the phase and token
        # progress.
        if members and (action in {"run_batch", "host_read"} or owners & {"preparation", "result", "publication"}):
            phase = members[0]["phase"]
            if phase in {"prefill", "decode"} and all(member["phase"] == phase for member in members):
                return phase, members

        context = query.get("context", {})
        if action == "batch_transform" and context.get("left") and context.get("operation") in {"filter", "merge"}:
            return "selection", [*context["left"], *context.get("right", [])]
        consumer = condition.get("read", {}).get("consumer")

        if (
            action == "query"
            and consumer == f"{symbols().req_time_stats}.{DecisionClocks.wait_queue_entry}"
            and condition.get("query") == "clock"
            and context.get("waiting")
            and all(context.get(key) == [] for key in ("running", "previous", "chunked"))
        ):
            return "admission", context["waiting"]

        # Nested decisions carry the queue at entry to native selection. A request may also be in the previous batch.
        if (queues := cls.queue_input(query)) is not None:
            members = {}
            for queue in queues.values():
                for member in queue:
                    if members.setdefault(member["id"], member) != member:
                        raise ValueError("native queues give different progress for the same request")

            return "scheduler", list(members.values())

    @classmethod
    def bind(cls, query, serving, domain):
        cut = cls.cut(query)
        if cut is None:
            return False

        mode = domain["implementation"]["disaggregation_mode"]
        instance, = (item for item in serving.instances if item.mode == mode)
        content = WorkloadContent(serving.workload["requests"], serving.workload["prefixes"])
        rows = []
        _, members = cut

        for member in members:
            identity = member["id"]
            rows.append(
                {
                    "id": identity,
                    "prompt": content.prompt_runs(identity),
                    "next_token": content.output_token(identity),
                    "output_tokens": content.requests[identity]["output_tokens"],
                }
            )

        query["native_inputs"] = {"configuration": instance.configuration.values, "requests": rows}
        return True


class RequestSubmissions:
    observed = "capture"

    def __init__(self, workload):
        self.requests = workload["requests"]
        self.content = WorkloadContent(self.requests, workload["prefixes"])

    @staticmethod
    def native_id(phase, index):
        return f"{phase}/{index}"

    @staticmethod
    def index(identity):
        return int(identity.rsplit("/", 1)[1])

    def cohorts(self, indices):
        arrivals = defaultdict(list)
        for index in indices:
            arrivals[self.requests[index]["arrival_ns"]].append(index)
        return sorted(arrivals.items())

    def sampling(self, index):
        return {"max_new_tokens": self.requests[index]["output_tokens"], "temperature": 0, "ignore_eos": True}

    @staticmethod
    def tokenized(identity, tokens, sampling, vocab_size, mode, *, extra_key=None, room=None):
        from array import array

        TokenizedGenerateReqInput = resolve(symbols().tokenized_generate_req_input)
        APIServerReqTimeStats = resolve(symbols().api_server_req_time_stats)
        SamplingParams = resolve(symbols().sampling_params)

        parameters = SamplingParams(**sampling)
        parameters.normalize(None)
        parameters.verify(vocab_size)
        return TokenizedGenerateReqInput(
            rid=identity, input_text=None, input_ids=array("q", tokens), input_embeds=None, mm_inputs=None,
            token_type_ids=None, sampling_params=parameters, return_logprob=False, logprob_start_len=-1,
            top_logprobs_num=0, token_ids_logprob=None, stream=True, extra_key=extra_key,
            bootstrap_host="prologue-prefill" if room is not None else None, bootstrap_room=room,
            time_stats=APIServerReqTimeStats(disagg_mode=mode),
        )

    def payload(self, phase, indices, prompts):
        return {
            "input_ids": [prompts[index] for index in indices],
            "sampling_params": [self.sampling(index) for index in indices],
            "rid": [self.native_id(phase, index) for index in indices],
            "stream": True,
        }


@dataclass
class StateRecord:
    members: dict
    values: list | None


class RequestStateCalls:
    fields = ("output_token", "metadata")

    def __init__(self, observe=False) -> None:
        self.observe = observe
        self.batch = ContextVar("state_batch", default=None)
        self.receiving = ContextVar("metadata_receiving", default=None)
        self.accesses = ContextVar("request_state_accesses", default=None)

    @contextmanager
    def members(self, batch):
        previous = self.batch.set(batch)
        try:
            yield
        finally:
            self.batch.reset(previous)

    @contextmanager
    def record(self):
        access = StateRecord(
            {f"{field}/{mode}": [] for field in self.fields for mode in ("read", "write")},
            [] if self.observe else None,
        )
        previous = self.accesses.set(access)
        try:
            yield access
        finally:
            self.accesses.reset(previous)

    def note(self, field, direction, requests):
        access = self.accesses.get()
        if access is not None:
            key = f"{field}/{direction}"
            access.members[key] = list(dict.fromkeys([*access.members[key], *(req.rid for req in requests)]))

    @classmethod
    def condition(cls, accesses):
        return sorted(
            f"{field}/{mode}" for field in cls.fields for mode in ("read", "write") if accesses.get(f"{field}/{mode}")
        )

    @contextmanager
    def receive(self, decode_req):
        previous = self.receiving.set(decode_req)
        try:
            yield
        finally:
            self.receiving.reset(previous)

    def set_metadata(self, original, buffers, req):
        self.note("metadata", "write", [req])
        self.metadata_values("write", req, req.output_ids[0])
        return original(buffers, req)

    def metadata_values(self, operation, req, token):
        record = self.accesses.get()
        if record is None or record.values is None:
            return

        # These Python values enter set_buf or have already been read by native commit; never synchronize a device for
        # diagnostics.
        record.values.append(
            {
                "field": "metadata",
                "operation": operation,
                "request": req.rid,
                "room": req.bootstrap_room,
                "values": None
                if token is None
                else {
                    "output_token": token,
                    **{
                        name: getattr(req, name)
                        for name in (
                            "cached_tokens",
                            "cached_tokens_device",
                            "cached_tokens_host",
                            "cached_tokens_storage",
                        )
                    },
                },
            }
        )

    def get_metadata(self, original, buffers, index):
        if self.accesses.get() is not None:
            request = self.receiving.get()
            if request is None or request.metadata_buffer_index != index:
                raise ValueError("native metadata read has no matching request reservation")
            self.note("metadata", "read", [request.req])
        return original(buffers, index)

    def stash(self, original, relay, indices, payload):
        if self.accesses.get() is not None and indices.shape[0]:
            batch = self.batch.get()
            if batch is None or indices is not batch.req_pool_indices or not relay.spec_algo.is_none():
                raise ValueError("native relay write has no matching non-speculative request binding")
            self.note("output_token", "write", batch.reqs)
        return original(relay, indices, payload)

    def resolve(self, original, batch, relay):

        # These are the native read conditions before resolve_forward_inputs mutates the batch.
        if batch.prefill_input_ids_cpu is not None:
            if batch.mix_running_indices is not None:
                self.note("output_token", "read", batch.decoding_reqs)
        elif batch.input_ids is None and relay.spec_algo.is_none():
            self.note("output_token", "read", batch.reqs)
        return original(batch, relay)

    def install(self):
        HookTable.around(symbols().future_map_stash, self.stash)
        HookTable.around(symbols().resolve_forward_inputs, self.resolve)
        HookTable.around(symbols().metadata_buffers_set_buf, self.set_metadata)
        HookTable.around(symbols().metadata_buffers_get_buf, self.get_metadata)
