"""Measure declared Decode work at a portable source cut, without serving requests."""

import json
from math import prod

from prologue.common.cost import ServiceSamples
from prologue.host.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.acquisition.local_base import LocalRule
from prologue.integrations.frameworks.sglang.acquisition.placement import SGLangAscendPlacement
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration
from prologue.integrations.platforms.ascend.host.deployment import AscendDeployment


class LocalAttention(LocalRule):
    name = "sglang.ascend.decode_attention"

    def __init__(self, sampling_sources):
        self.sampling_sources = sampling_sources

    @staticmethod
    def matches(query):
        if query.get("work") != "device" or query["features"]["phase"] != "decode" or "transport" in query["features"]:
            return False
        definition, domain = query["definition"], query["domain"]
        parallel = domain["parallel"]
        return (domain["framework"] == "sglang" and "torch_npu" in domain and not domain["expert_layers"]
                and domain["implementation"]["disaggregation_mode"] == "null"
                and parallel["logical_tp"] == parallel["logical_pp"] == parallel["logical_ep"] == 1
                and definition.get("role") == "attention" and not definition["children"])

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query["definition"]

    def prepare(self, queries, deployment, arguments, limit):
        domain, definition = self.common_session(queries, deployment, "worker", "attention_cut")
        contexts = set()
        for q in queries:
            lengths = LocalAttention.contexts(q["features"]["axes"])
            if not lengths:
                raise EvidenceGap("unbound_decode_context", features=q["features"])
            contexts.add(lengths)

        selected = None
        if self.sampling_sources is None:
            raise EvidenceGap("missing_qualified_attention_cut", contexts=sorted(contexts))
        for recording, source in self.sampling_sources():
            if source["domain"] != domain:
                continue
            settings = NativeConfiguration(source["runner"]["configuration"])
            if not all(LocalAttention.fits(settings, batch) for batch in contexts):
                continue
            for cut in source["cuts"]:
                if cut["definition"] != definition:
                    continue
                if selected and selected[2]["parts"] != cut["parts"]:
                    raise EvidenceGap("ambiguous_device_partition")
                if selected is None:
                    selected = recording, source["runner"], cut
        if selected is None:
            raise EvidenceGap("missing_qualified_attention_cut", contexts=sorted(contexts))
        recording, runner, cut = selected
        platform = AscendDeployment(deployment, 1, False)
        revision = domain["implementation"]["framework_revision"]
        job = {"runner": runner, "site": cut["site"], "contexts": [list(batch) for batch in sorted(contexts)]}
        return SGLangAscendPlacement.local_attempt(arguments, limit, deployment, platform.devices, revision, job,
                                                "--local-attention", runtime_trace=False, model=runner["model"]["path"],
                                                template_recording=recording, cut=cut, domain=domain)

    @staticmethod
    def committed(call, lengths):
        if "kv" not in call:
            return False
        prefill, decode = call["kv"]["prefill"], call["kv"]["decode"]
        return (all(row["matching_operands"] and row["preserved_prefix"] for row in (prefill, decode))
                and prefill["slots"] == decode["slots"] and prefill["committed"] == list(lengths)
                and decode["committed"] == [n + 1 for n in lengths]
                and prefill["locations"] == [row[:-1] for row in decode["locations"]]
                and prefill["written"] == [loc for row in prefill["locations"] for loc in row]
                and decode["written"] == [row[-1] for row in decode["locations"]])

    @classmethod
    def qualify(cls, queries, recording, timeout=None):
        inputs = recording.inputs
        domain, cut = inputs["domain"], inputs["cut"]
        record = json.loads((recording.source / recording.manifest["measurements"]).read_bytes())
        native = domain["implementation"]
        same_worker = (record["rank"] == WorkerDomain.rank(domain) and record["load_format"] == native["load_format"]
                       and record["model"] == native["model"]
                       and all(record["runtime"][key] == domain[key] for key in ("torch", "torch_npu", "device")))
        samples, measured, answers = ServiceSamples(domain, [cut["definition"]]), {}, []
        published = set()
        for q in queries:
            lengths = LocalAttention.contexts(q["features"]["axes"])
            if lengths not in measured:
                calls = [
                    call
                    for call in record["calls"]
                    if call["site"] == cut["site"] and tuple(call["context"]) == lengths
                ]
                costs, detail = None, "unobserved_worker_or_kv"
                if same_worker and calls and all(cls.committed(call, lengths) for call in calls):
                    costs, detail = LocalAttention.measured(domain, cut["definition"], cut["parts"], calls, lengths)
                measured[lengths] = costs, detail
            costs, detail = measured[lengths]
            if (not cls.matches(q) or q["domain"] != domain or q["definition"] != cut["definition"]
                    or costs is None or q["interval"] not in costs):
                answers.append({"status": "untriggered", "detail": detail})
                continue

            # Multiple requesting targets can share a coordinate. The measured interval contributes one observation.
            key = lengths, q["interval"]
            if key not in published:
                samples.add(0, q["interval"], q["features"], costs[q["interval"]])
                published.add(key)
            answers.append({"status": "qualified", "asset": "samples/0.json", "observations": 1})
        products = {}
        if samples.rows:
            products["samples/0.json"] = samples.document({"site": cut["site"], "measurements": [
                {"contexts": lengths, "result": detail} for lengths, (_, detail) in measured.items()
            ]})
        return products, answers

    @staticmethod
    def contexts(axes):
        members = axes.get("batch.members", 0)
        maximum = axes.get("batch.max_context_tokens", 0)
        total = axes.get("batch.context_tokens", 0)
        if any(n < 1 or n != int(n) for n in (members, maximum, total)):
            return ()
        members, maximum, total = map(int, (members, maximum, total))

        # Only invert coordinates that determine the actual multiset of lengths for unit Decode.
        if total == maximum * members:
            lengths = (maximum,) * members
        elif members == 2 and 0 < total - maximum <= maximum:
            lengths = (total - maximum, maximum)
        else:
            return ()
        known = {"batch.members": members, "batch.query_tokens": members, "batch.max_query_tokens": 1,
                 "batch.context_tokens": total, "batch.max_context_tokens": maximum,
                 "batch.attention_pairs": total + members, "parameter.model_tokens": members}
        return lengths if axes == known else ()

    @staticmethod
    def fits(settings, lengths):
        page = settings.values["page_size"]
        return (bool(lengths) and len(lengths) <= settings.batch_capacity(settings.values)
                and sum(lengths) <= settings.values["max_prefill_tokens"]
                and sum((length // page + 1) * page for length in lengths) <= settings.values["max_total_tokens"])

    @staticmethod
    def same_heads(call, definition):
        heads = call["heads"]
        widths = {"argument.q": heads["query"] * heads["key_dim"], "argument.k": heads["key"] * heads["key_dim"],
                  "argument.v": heads["value"] * heads["value_dim"]}
        seen = set()
        for port in definition["ports"]:
            name = port["name"]
            if name not in widths or port["direction"] != "input":
                continue
            shape = port["tensor"]["shape"]
            if (
                not shape
                or shape[0] != "model_tokens"
                or not all(type(x) is int for x in shape[1:])
                or prod(shape[1:]) != widths[name]
                or call["operands"].get(name)
                != {"shape": [call["query_rows"], *shape[1:]], "dtype": port["tensor"]["dtype"]}
            ):
                return False
            seen.add(name)
        return seen == widths.keys()

    @staticmethod
    def measured(domain, definition, parts, calls, lengths):
        if not parts:
            return None, "source_has_no_single_stream_partition"
        matching = [c for c in calls if tuple(sorted(c["context"])) == lengths and c["query_rows"] == len(lengths)]
        if len(matching) != 2 or "profile" in matching[0] or "profile" not in matching[1]:
            return None, "native_batch_geometry_not_repeated"
        plain, measured = matching
        native = domain["implementation"]
        for call in matching:
            if (call["kv_layout"] != domain["kv_layout"] or call["backend"] != native["attention_backend"]
                    or call["implementation"] != definition["implementation"]
                    or call["page_size"] != native["page_size"]
                    or not call["full_context"]
                    or not LocalAttention.same_heads(call, definition)):
                return None, "attention_domain_or_read_extent_changed"
        if "reason" in measured:
            return None, measured["reason"]
        rows = measured["tasks"]
        if [tuple(task[:2]) for task in rows] != [tuple(task) for _, tasks in parts for task in tasks]:
            return None, "device_task_sequence_changed"
        durations = iter(row[2] for row in rows)
        costs = {interval: sum(next(durations) for _ in tasks) for interval, tasks in parts}
        return costs, {"events_only_ns": plain["envelope_ns"], "with_profiler_ns": measured["envelope_ns"]}
