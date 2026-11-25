"""Acquire the native tensor-parallel input embedding at the requested token count."""

import json

from prologue.acquisition.rules import EvidenceGap, missing_root
from prologue.integrations.frameworks.sglang.acquisition.local_base import LocalRule
from prologue.integrations.frameworks.sglang.acquisition.placement import SGLangAscendPlacement
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.deployment import AscendDeployment


class LocalEmbedding(LocalRule):
    name = "sglang.vocab_parallel_embedding"

    @staticmethod
    def matches(query):
        if query.get("action") != "embedding":
            return False
        domain, condition = query["domain"], query["condition"]
        parallel, impl = domain["parallel"], domain["implementation"]
        return (missing_root(query)
                and set(query["parameters"]) == {"model_tokens"}
                and condition.keys() == {"phase", "state_accesses"} and condition["state_accesses"] == []
                and condition["phase"] in {"prefill", "decode", "mixed"}
                and domain["framework"] == "sglang" and "torch_npu" in domain
                and parallel["logical_tp"] > 1 and parallel["logical_pp"] == parallel["logical_ep"] == 1
                and impl["disaggregation_mode"] == "null" and impl["load_format"] == "dummy"
                and not impl["model"].get("quantization_config") and not domain["expert_layers"])

    @staticmethod
    def session_key(query, deployment):
        common = {k: v for k, v in query["domain"].items() if k not in WorkerDomain.role_fields}
        return common, query["condition"], query["parameters"]

    @classmethod
    def prepare(cls, queries, deployment, arguments, limit):
        first = queries[0]
        domain, count = first["domain"], first["parameters"]["model_tokens"]
        if type(count) is not int or count <= 0:
            raise EvidenceGap("unbound_token_count", required="positive_flat_token_count")
        cls.common_session(queries, deployment, "model_domain", "embedding_call")
        ranks = set()
        for query in queries:
            other = query["domain"]
            rank = WorkerDomain.rank(other)
            role = domain["worker_roles"].get(str(rank))
            requested = other | {"parallel": {**other["parallel"], "rank": rank}}
            if role is None or WorkerDomain.bind_worker(domain, role) != requested:
                raise EvidenceGap("different_sessions_required", conditions=["model_domain", "embedding_call"])
            ranks.add(rank)
        placement = AscendDeployment(deployment, domain["parallel"]["logical_tp"], False)
        impl = domain["implementation"]
        job = {"domain": domain, "ranks": sorted(ranks), "phase": first["condition"]["phase"],
               "tokens": count, "single_device": placement.single_device}
        return SGLangAscendPlacement.local_attempt(
            arguments, limit, deployment, placement.devices, impl["framework_revision"], job, "--local-embedding",
            runtime_trace=True, model=impl["model"]["_name_or_path"],
        )

    @staticmethod
    def check_call(facts, readback, query, job, rank):
        domain = query["domain"]
        env = dict(facts["environment"])
        device = env.pop("device_index")
        parallel = dict(env["parallel"])
        observed_rank = parallel.pop("rank")
        env["parallel"] = parallel
        if env != domain or observed_rank != rank or device != (0 if job["single_device"] else rank):
            raise ValueError("embedding evidence belongs to another worker or runtime")

        scopes = facts["scopes"]
        root, = (i for i, s in enumerate(scopes) if s["parent"] == 0 and s["role"] == "embedding")
        child, = (i for i, s in enumerate(scopes) if s["parent"] == root and s["role"] == "communication")
        call, reduce = scopes[root], scopes[child]
        exchange = reduce["communication"]
        group = domain["worker_roles"][str(rank)]["groups"][reduce["bindings"]["collective_group"]]
        ports = {p["name"]: p["tensor"] for p in call["ports"] if p.get("tensor")}
        tokens, result = ports["argument.input_"], ports["result"]
        shape = [job["tokens"], domain["implementation"]["model"]["hidden_size"]]
        if (call["parameters"] != query["parameters"] or call["work"]["phase"] != query["condition"]["phase"]
                or readback["site"] != call["site"] or not readback["matched_owning_weights"]
                or tokens["shape"] != [job["tokens"]] or readback["shape"] != result["shape"]
                or readback["dtype"] != result["dtype"] or result["shape"] != shape
                or exchange["operation"] != "all_reduce" or exchange["rank"] != rank
                or exchange["members"] != group["members"] or exchange["required"] != group["members"]):
            raise ValueError("embedding call, weight readback and native peer contract disagree")

        # Native HCCL can divide one logical reduction over several physical stream intervals.
        services = [s for s in facts["spans"] if s["scope"] == child and s["kind"] == "collective"]
        if not services:
            raise ValueError("embedding reduction lacks an observed peer service")
        for service in services:
            lane = facts["resources"][service["resource"]]
            if job["single_device"]:
                operand = service["protocol_operand"]
                if (
                    lane["kind"] != "service"
                    or operand["operation"] != "all_reduce"
                    or operand["dtype"] != result["dtype"]
                ):
                    raise ValueError("staged embedding reduction lacks its recorded peer operand")
            elif lane["kind"] != "stream" or "protocol_operand" in service:
                raise ValueError("physical embedding reduction lacks an observed device service")

    @classmethod
    def qualify(cls, queries, recording, timeout=None):
        job = recording.inputs["job"]
        assets, answers = {}, []
        for query in queries:
            domain = query["domain"]
            rank = WorkerDomain.rank(domain)
            asset = f"structure/embedding-{rank}.json"
            if asset not in assets:
                native = recording.manifest["workers"][str(rank)]
                facts = json.loads((recording.source / native["facts"]).read_bytes())
                readback = json.loads((recording.source / native["readback"]).read_bytes())
                cls.check_call(facts, readback, query, job, rank)
                assets[asset] = facts
            answers.append({"status": "qualified", "asset": asset, "rank": rank,
                            "transport": "staged_gloo" if job["single_device"] else "physical_hccl"})
        return assets, answers
