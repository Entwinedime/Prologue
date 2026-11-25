"""Construct the SGLang input cohorts shared by observation and control."""

from collections import defaultdict

from prologue.common.workload import WorkloadContent
from prologue.integrations.frameworks.sglang.shared.clocks import DecisionClocks


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
            and consumer == f"sglang.srt.observability.req_time_stats.{DecisionClocks.wait_queue_entry}"
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

        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
        from sglang.srt.sampling.sampling_params import SamplingParams

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
