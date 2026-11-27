"""Acquire receiver table and vocabulary cuts through their native worker entry points."""

from collections import defaultdict

from prologue.common.cost import ServiceSamples, cost_features, cost_identity
from prologue.common.io import read_json
from prologue.host.acquisition.rules import EvidenceGap
from prologue.host.libraries.release import asset_path
from prologue.integrations.bindings import RuleName, device_runtime, platform_backend
from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors

from ..contracts import CopyOperand, LocalCut, LocalRule, WorkerDomain, embedding_rule
from ..endpoints import copy_sizes


class LocalPrebuilt(LocalRule):
    name = "sglang.pd_receiver_prebuilt_table"
    worker = "--local-prebuilt"
    conditions = ("worker", "pool_geometry")
    condition = {"owner": "prebuilt_prepare", "entry": "call.enter", "exit": "call.return",
                 "phase": "prefill", "state_accesses": []}
    capacity = ("page_size", "kv_pages", "slot_capacity", "sequence_capacity")

    def matches(self, query):
        return (self.missing_action(query, "segment") and query["condition"] == self.condition
                and query["domain"]["implementation"]["disaggregation_mode"] == "decode")

    def operands(self, query):
        members = query.get("members", [])
        params = query["parameters"]
        lengths = [m["new_tokens"] for m in members]

        if (not members or len({m["id"] for m in members}) != len(members)
                or any(m["context_tokens"] != 0 for m in members)
                or any(type(n) is not int or n <= 0 for n in lengths)):
            raise EvidenceGap("unbound_admitted_members", required="ordered_distinct_members_with_cold_local_kv")
        if ((len(lengths), sum(lengths)) != (params["members"], params["input_tokens"])
                or max(lengths) > params["sequence_capacity"] or len(lengths) > params["slot_capacity"]
                or sum((n + params["page_size"] - 1) // params["page_size"] for n in lengths) > params["kv_pages"]):
            raise EvidenceGap("incompatible_receiver_geometry", lengths=lengths, parameters=params)

        return members

    @classmethod
    def session_key(cls, query, deployment):
        return query["domain"], {k: query["parameters"][k] for k in cls.capacity}

    def cut(self, queries, session, deployment, arguments):
        domain, capacity = session
        if any(type(value) is not int or value <= 0 for value in capacity.values()):
            raise ValueError("receiver pool capacities must be positive integers")

        impl = domain["implementation"]
        if (not impl["cache"]["disabled"] or capacity["page_size"] != impl["page_size"]
                or impl["allocator"] != symbols().paged_allocator
                or impl["cache"]["type"] != symbols().chunk_cache):
            raise EvidenceGap("unsupported_receiver_pool", allocator=impl["allocator"], cache=impl["cache"])

        batches = [self.operands(q) for q in queries]
        page = capacity["page_size"]
        sizes = [m["new_tokens"] for batch in batches for m in batch]
        if len(sizes) > capacity["slot_capacity"] or sum((s + page - 1) // page for s in sizes) > capacity["kv_pages"]:
            raise EvidenceGap("different_sessions_required", conditions=["live_receiver_slots_and_pages"])

        job = {"domain": domain, "rank": WorkerDomain.rank(domain), "capacity": capacity, "batches": batches}
        return LocalCut(job, [self.physical_card(domain, deployment)], impl["framework_revision"],
                        impl["model"]["_name_or_path"])

    def qualify(self, queries, recording, timeout=None):
        answers = []
        facts, measurements = (self.recording_part(recording, part) for part in ("facts", "measurements"))
        environment, rank, _ = WorkerDomain.local_environment(facts["environment"])
        calls = [scope for scope in facts["scopes"] if scope["role"] == "segment"
                 and scope["bindings"]["segment"]["owner"] == self.condition["owner"]]

        for query, row, call in zip(queries, measurements, calls, strict=True):
            members = self.operands(query)
            if environment != query["domain"] or rank != WorkerDomain.rank(query["domain"]):
                raise ValueError("receiver evidence belongs to another worker or runtime")

            work = call["work"]
            outputs = {p["name"] for p in call["ports"] if p["direction"] == "output"}
            if (work["members"] != members or call["parameters"] != query["parameters"]
                    or work["phase"] != self.condition["phase"]
                    or any(work["continuation"][k] != self.condition[k] for k in ("entry", "exit"))
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

            BatchTensors.bind_location_rows(call, row["request_slots"], row["writes"])
            answers.append({"status": "qualified", "asset": "structure/receiver.json"})

        # One observation contains all calls. A failed readback would make a shared source unsafe to import.
        if any(answer["status"] != "qualified" for answer in answers):
            incomplete = {"status": "untriggered", "missing": "verified_session"}
            return {}, [answer if answer["status"] != "qualified" else incomplete for answer in answers]
        return {"structure/receiver.json": facts}, answers


class LocalEmbedding(LocalRule):
    name = embedding_rule
    worker = "--local-embedding"
    conditions = ("model_domain", "embedding_call")

    def matches(self, query):
        if not self.missing_action(query, "embedding"):
            return False
        domain, condition = query["domain"], query["condition"]
        impl = domain["implementation"]
        return (set(query["parameters"]) == {"model_tokens"}
                and condition.keys() == {"phase", "state_accesses"} and condition["state_accesses"] == []
                and condition["phase"] in {"prefill", "decode", "mixed"}
                and WorkerDomain.dense_worker(domain) and domain["parallel"]["logical_tp"] > 1
                and impl["load_format"] == "dummy" and not impl["model"].get("quantization_config"))

    @staticmethod
    def session_key(query, deployment):
        return WorkerDomain.role_plan(query["domain"]), query["condition"], query["parameters"]

    def cut(self, queries, session, deployment, arguments):
        first = queries[0]
        domain, count = first["domain"], first["parameters"]["model_tokens"]
        if type(count) is not int or count <= 0:
            raise EvidenceGap("unbound_token_count", required="positive_flat_token_count")

        ranks = set()
        for query in queries:
            other = query["domain"]
            rank = WorkerDomain.rank(other)
            role = domain["worker_roles"].get(str(rank))
            requested = other | {"parallel": {**other["parallel"], "rank": rank}}

            if role is None or WorkerDomain.bind_worker(domain, role) != requested:
                raise EvidenceGap("different_sessions_required", conditions=["model_domain", "embedding_call"])
            ranks.add(rank)

        placement = platform_backend().deployment(deployment, domain["parallel"]["logical_tp"], False)
        impl = domain["implementation"]
        job = {"domain": domain, "ranks": sorted(ranks), "phase": first["condition"]["phase"],
               "tokens": count, "single_device": placement.single_device}
        return LocalCut(job, placement.devices, impl["framework_revision"], impl["model"]["_name_or_path"])

    def check_call(self, facts, readback, query, job, rank):
        domain = query["domain"]
        env, observed_rank, device = WorkerDomain.local_environment(facts["environment"])
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

        # A native collective can divide one logical reduction over several physical stream intervals.
        services = [s for s in facts["spans"] if s["scope"] == child and s["kind"] == "collective"]
        if not services:
            raise ValueError("embedding reduction lacks an observed peer service")
        for service in services:
            lane = facts["resources"][service["resource"]]
            if job["single_device"]:
                operand = service["protocol_operand"]
                if (lane["kind"], operand["operation"], operand["dtype"]) != ("service", "all_reduce", result["dtype"]):
                    raise ValueError("staged embedding reduction lacks its recorded peer operand")
            elif lane["kind"] != "stream" or "protocol_operand" in service:
                raise ValueError("physical embedding reduction lacks an observed device service")

    def qualify(self, queries, recording, timeout=None):
        job = recording.inputs["job"]
        assets, answers = {}, []
        for query in queries:
            domain = query["domain"]
            rank = WorkerDomain.rank(domain)
            asset = f"structure/embedding-{rank}.json"

            if asset not in assets:
                native = recording.manifest["workers"][str(rank)]
                facts = read_json(asset_path(recording.source, native["facts"]))
                readback = read_json(asset_path(recording.source, native["readback"]))
                self.check_call(facts, readback, query, job, rank)
                assets[asset] = facts

            answers.append({"status": "qualified", "asset": asset, "rank": rank})

        return assets, answers


class LocalCopies(CopyOperand, LocalRule):
    name = RuleName("synchronous_tensor_copy", platform_only=True)
    worker = "--local-copy"
    priority = 10
    source_window = True
    conditions = ("physical_device", "framework_revision", "runtime")

    @classmethod
    def session_key(cls, query, deployment):
        domain = query["domain"]
        card = cls.physical_card(domain, deployment)
        return card, domain["implementation"]["framework_revision"], cls.runtime(domain)

    def cut(self, queries, session, deployment, arguments):
        card, revision, runtime = session
        operands = defaultdict(lambda: defaultdict(set))
        for query in queries:
            caller, dtype, width, size = self.point(query)
            for point in copy_sizes(query, width, size):
                operands[caller, dtype, width][point].add(WorkerDomain.rank(query["domain"]))

        cuts = [{"caller": caller, "dtype": dtype, "element_bytes": width,
                 "points": [{"bytes": size, "ranks": sorted(ranks)} for size, ranks in sorted(sizes.items())]}
                for (caller, dtype, width), sizes in sorted(operands.items())]
        return LocalCut({"runtime": runtime, "operands": cuts}, [card], revision)

    def measurements(self, recording):
        manifest, job = recording.manifest, recording.inputs["job"]
        services = {op["caller"]: device_runtime().copy_sampler.service(op["caller"]) for op in job["operands"]}
        if (manifest["completion"] != "successful_synchronous_acl_return" or manifest["services"] != services
                or any(manifest["runtime"][k] != v for k, v in job["runtime"].items())):
            raise ValueError("native copy observation has different runtime, completion or service ownership")

        planned = {(op["caller"], op["dtype"], p["bytes"]): (op["element_bytes"], set(p["ranks"]))
                   for op in job["operands"] for p in op["points"]}
        if len(planned) != sum(len(op["points"]) for op in job["operands"]):
            raise ValueError("a native copy point must have one element width and one rank set")

        observed = self.recording_part(recording, "measurements")
        rows = {(r["caller"], r["dtype"], r["bytes"]): r for r in observed}
        if len(rows) != len(observed) or rows.keys() - planned.keys():
            raise ValueError("native copy recording has duplicate or unrequested operands")

        return planned, rows

    def qualify(self, queries, recording, timeout=None):
        planned, rows = self.measurements(recording)
        inputs = recording.inputs
        groups, answers = {}, []

        for query in queries:
            domain = query["domain"]
            caller, dtype, width, size = self.point(query)
            key = caller, dtype, size
            rank = WorkerDomain.rank(domain)
            verified = {
                point: rows[point] for n in copy_sizes(query, width, size) if (point := (caller, dtype, n)) in rows
                and planned[point][0] == width and rank in planned[point][1]
                and rows[point]["verified_operand"] and rows[point]["observations_ns"]
            }

            if (domain["implementation"]["framework_revision"] != inputs["framework_revision"]
                    or self.runtime(domain) != self.runtime(recording.manifest["runtime"])):
                answers.append({"status": "untriggered", "missing": "target_runtime"})
            elif key not in verified:
                answers.append({"status": "untriggered", "missing": "matching_synchronous_copy_and_destination"})
            else:
                name, _, keys = groups.setdefault(cost_identity(domain), (f"samples/{len(groups)}.json", domain, set()))
                keys.update(verified)
                answer = {"status": "qualified", "asset": name, "observations": len(verified[key]["observations_ns"])}
                if endpoints := sorted(n for _, _, n in verified if n != size):
                    answer["endpoint_bytes"] = endpoints

                answers.append(answer)

        products = {}
        for name, domain, keys in groups.values():
            samples, unmatched = ServiceSamples(domain), 0
            for key in sorted(keys):
                row = rows[key]
                unmatched += row["unmatched_calls"]
                service = device_runtime().copy_sampler.service(row["caller"])
                features = cost_features({"transfer.bytes": row["bytes"]})

                for duration in row["observations_ns"]:
                    samples.add_external(service, "transfer", features, duration)

            products[name] = samples.document({"unmatched_calls": unmatched, "unverified_operands": 0})

        return products, answers


rules = (LocalPrebuilt, LocalEmbedding, LocalCopies)
