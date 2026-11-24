"""Acquire the local PD receiver table for the controller's ordered, admitted requests."""

import json

from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.deployment import AscendDeployment


class LocalPrebuilt:
    name = "sglang.pd_receiver_prebuilt_table"
    condition = {"owner": "prebuilt_prepare", "entry": "call.enter", "exit": "call.return",
                 "phase": "prefill", "state_accesses": []}
    capacity = ("page_size", "kv_pages", "slot_capacity", "sequence_capacity")

    @classmethod
    def matches(cls, query):
        domain = query["domain"]
        return (query.get("status") in {"no_declared_candidates", "unmatched_conditions"}
                and query.get("action") == "segment" and query["condition"] == cls.condition
                and domain["framework"] == "sglang" and "torch_npu" in domain
                and domain["implementation"]["disaggregation_mode"] == "decode")

    @classmethod
    def operands(cls, query):
        members = query.get("members", [])
        params = query["parameters"]
        if (not members or len({m["id"] for m in members}) != len(members)
                or any(m["context_tokens"] != 0 or type(m["new_tokens"]) is not int or m["new_tokens"] <= 0
                       for m in members)):
            raise EvidenceGap("unbound_admitted_members", required="ordered_distinct_members_with_cold_local_kv")
        lengths = [m["new_tokens"] for m in members]
        if ((len(lengths), sum(lengths)) != (params["members"], params["input_tokens"])
                or max(lengths) > params["sequence_capacity"] or len(lengths) > params["slot_capacity"]
                or sum((n + params["page_size"] - 1) // params["page_size"] for n in lengths) > params["kv_pages"]):
            raise EvidenceGap("incompatible_receiver_geometry", lengths=lengths, parameters=params)
        return members

    @classmethod
    def prepare(cls, queries, deployment, arguments, limit):
        first = queries[0]
        domain = first["domain"]
        capacity = {k: first["parameters"][k] for k in cls.capacity}
        if any(type(value) is not int or value <= 0 for value in capacity.values()):
            raise ValueError("receiver pool capacities must be positive integers")
        if any(q["domain"] != domain or any(q["parameters"][k] != v for k, v in capacity.items()) for q in queries):
            raise EvidenceGap("different_sessions_required", conditions=["worker", "pool_geometry"])
        impl = domain["implementation"]
        if (not impl["cache"]["disabled"] or capacity["page_size"] != impl["page_size"]
                or impl["allocator"] != "sglang.srt.hardware_backend.npu.allocator_npu.NPUPagedTokenToKVPoolAllocator"
                or impl["cache"]["type"] != "sglang.srt.mem_cache.chunk_cache.ChunkCache"):
            raise EvidenceGap("unsupported_receiver_pool", allocator=impl["allocator"], cache=impl["cache"])
        batches = [cls.operands(q) for q in queries]
        page = capacity["page_size"]
        sizes = [m["new_tokens"] for batch in batches for m in batch]
        if len(sizes) > capacity["slot_capacity"] or sum((s + page - 1) // page for s in sizes) > capacity["kv_pages"]:
            raise EvidenceGap("different_sessions_required", conditions=["live_receiver_slots_and_pages"])
        rank = WorkerDomain.rank(domain)
        placement = AscendDeployment(deployment, max(map(int, domain["worker_roles"])) + 1, False)
        job = {"domain": domain, "rank": rank, "capacity": capacity, "batches": batches}
        card = placement.devices[0 if placement.single_device else rank]
        return SGLangAscendLaunch.local_attempt(
            arguments, limit, deployment, card, impl["framework_revision"], job, "--local-prebuilt",
            runtime_trace=True, model=impl["model"]["_name_or_path"],
        )

    @classmethod
    def qualify(cls, queries, recording):
        answers = []
        measurements = json.loads((recording.source / recording.manifest["measurements"]).read_bytes())
        facts = json.loads((recording.source / recording.manifest["facts"]).read_bytes())
        environment = dict(facts["environment"])
        rank = environment["parallel"]["rank"]
        environment["parallel"] = {k: v for k, v in environment["parallel"].items() if k != "rank"}
        environment.pop("device_index")
        calls = [scope for scope in facts["scopes"] if scope["role"] == "segment"
                 and scope["bindings"]["segment"]["owner"] == cls.condition["owner"]]
        for query, row, call in zip(queries, measurements, calls, strict=True):
            members = cls.operands(query)
            if environment != query["domain"] or rank != WorkerDomain.rank(query["domain"]):
                raise ValueError("receiver evidence belongs to another worker or runtime")
            work = call["work"]
            outputs = {p["name"] for p in call["ports"] if p["direction"] == "output"}
            if (work["members"] != members or call["parameters"] != query["parameters"]
                    or work["phase"] != cls.condition["phase"]
                    or any(work["continuation"][k] != cls.condition[k] for k in ("entry", "exit"))
                    or not {"request_slots", "sequence_lengths", "cache_locations"} <= outputs):
                raise ValueError("receiver table readback and observed semantic call disagree")

            # Slot ordering and copy geometry must agree with both the query and the device readback.
            expected, offset = [], 0
            for member, slot in zip(members, row["request_slots"], strict=True):
                count = member["new_tokens"]
                expected.append({"slot": slot, "source": [0, count], "target": [offset, offset + count]})
                offset += count
            if (row["members"] != members or row["parameters"] != query["parameters"] or row["writes"] != expected
                    or len(set(row["request_slots"])) != len(members) or not row["verified_kv"]
                    or row["completion"] != "device_event_synchronized_before_consumer_read"):
                answers.append({"status": "untriggered", "missing": "matching_receiver_writes_and_readable_kv"})
                continue
            answers.append({"status": "qualified", "asset": "structure/receiver.json", "members": len(members),
                            "observed_writes": len(expected),
                            "boundary": "admitted_members_and_delivered_local_kv_to_readable_batch_outputs"})

        # One observation contains all calls. A failed readback would make a shared source unsafe to import.
        if any(answer["status"] != "qualified" for answer in answers):
            incomplete = {"status": "untriggered", "missing": "verified_session"}
            return {}, [answer if answer["status"] != "qualified" else incomplete for answer in answers]
        facts["receiver_readback"] = {
            "admission": "ordered_controller_input", "local_kv": "native_pool_seeded_before_call",
            "calls": [row | {"scope": call["bindings"]["source_scopes"][0]}
                      for row, call in zip(measurements, calls, strict=True)],
        }
        return {"structure/receiver.json": facts}, answers
