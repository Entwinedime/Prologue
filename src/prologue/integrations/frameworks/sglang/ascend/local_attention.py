"""Measure declared Decode work at a portable source cut, without serving requests."""

import json

from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.ascend.attention_cuts import AttentionCuts
from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.deployment import AscendDeployment
from prologue.performance.samples import ServiceSamples


class LocalAttention:
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

    def prepare(self, queries, deployment, arguments, limit):
        domain, definition = queries[0]["domain"], queries[0]["definition"]
        contexts = set()
        for q in queries:
            if q["domain"] != domain or q["definition"] != definition:
                raise EvidenceGap("different_sessions_required", conditions=["worker", "attention_cut"])
            lengths = AttentionCuts.contexts(q["features"]["axes"])
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
            if not all(AttentionCuts.fits(settings, batch) for batch in contexts):
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
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, platform.devices, revision, job,
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
    def qualify(cls, queries, recording):
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
            lengths = AttentionCuts.contexts(q["features"]["axes"])
            if lengths not in measured:
                calls = [call for call in record["calls"] if call["site"] == cut["site"] and tuple(call["context"]) == lengths]
                costs, detail = None, "unobserved_worker_or_kv"
                if same_worker and calls and all(cls.committed(call, lengths) for call in calls):
                    costs, detail = AttentionCuts.measured(domain, cut["definition"], cut["parts"], calls, lengths)
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
