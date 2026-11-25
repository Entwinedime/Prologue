"""Local native cuts for worker bindings, scheduler batches, embeddings, receiver tables, copies and attention."""

from collections import defaultdict
from itertools import pairwise
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import ServiceSamples, cost_features, cost_identity
from prologue.common.io import read_json, write_json
from prologue.host.acquisition.rules import EvidenceGap, missing_root
from prologue.host.libraries.release import asset_path
from prologue.integrations.bindings import RuleName, device_runtime, platform_backend, platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchOperands, BatchTensors
from prologue.integrations.frameworks.sglang.shared.configuration import AttentionCatalog, NativeConfiguration

from . import CopyOperand, LocalCut, LocalRule, WorkerDomain, embedding_rule, scheduler_batch_rule
from .evidence import BatchReadback, CostSamples, HostCostSamples, WorkerCapture


class LocalWorkerBinding(LocalRule):
    name = RuleName("worker_initialization")
    conditions = ("worker_initial_state",)

    def matches(self, query):
        if query.get("status") != "missing_worker_binding" or query.get("action") != "initialize_workers":
            return False
        structures = query["domain"]["instances"]
        structure = structures.get("null", {})
        return (structures.keys() == {"null"} and len(query["instances"]) == 1
                and structure["attention_backend"] in platform_domain().strided_attention
                and structure["pp"] == structure["ep"] == 1
                and "expert_layout" not in query["instances"][0]["readback"]["implementation"])

    def cut(self, queries, first, deployment, arguments):
        instance, = first["instances"]
        structure = first["domain"]["instances"]["null"]
        if (instance["mode"], instance["rank_begin"], instance["rank_end"]) != ("null", 0, structure["tp"]):
            raise ValueError("worker initialization requires one complete native TP group")
        settings = instance["settings"]
        configuration = NativeConfiguration({
            **settings, **{k: structure[k] for k in ("tp", "pp", "attention_backend", "prefix_cache", "mixed_batches")},
        })
        if configuration.structure() != structure or configuration.settings() != settings:
            raise ValueError("worker initialization conditions disagree with the requested structure")
        if settings["max_total_tokens"] < 2:
            raise EvidenceGap("insufficient_warmup_capacity", required_tokens=2)

        model = first["domain"]["model"]
        job = {"model": model, "configuration": configuration.values}
        return LocalCut.workers(job, deployment, structure["tp"])

    def initialized(self, worker, query):
        asset = worker.read("facts") | worker.read("binding")
        if worker.read("boundary") != {"phase": "idle", "members": []} or not any(
            s["bindings"].get("segment", {}).get("owner") == "selection" for s in asset["scopes"]
        ):
            raise ValueError("initialized worker did not execute its idle selection boundary")
        env, capacity, initial = asset["environment"], asset["capacity"], asset["initial_state"]
        rank = worker.rank
        role = env["worker_roles"][str(rank)]
        if (env["implementation"]["framework_revision"] != worker.recording.inputs["framework_revision"]
                or env["parallel"]["rank"] != rank or role["representative"] != rank or not role["modules"]
                or env["parallel"] != role["parallel"] or env["kv_layout"] != role["kv_layout"]):
            raise ValueError("native readback does not describe the initialized worker and its model")

        instance, = query["instances"]
        settings = instance["settings"]
        sections = {"environment": env, "implementation": env["implementation"], "worker_roles": role,
                    "initial_state": initial}
        if WorkerDomain.observed([{"domain": env}]) != query["domain"] or any(
            set(fields) - sections[key].keys() for key, fields in instance["readback"].items()
        ):
            raise ValueError("native worker does not satisfy the requested readback")
        warm = {"prefill_tokens": [1], "decode_steps": 1, "device_drained": True, "device_cache_emptied": False}
        if (capacity["page_size"] != settings["page_size"] or asset["configured_settings"] != settings
                or initial["logical_kv"] != "empty_after_native_flush" or initial["workspace"] != warm
                or initial["free_slots"] != capacity["slot_capacity"]
                or initial["free_tokens"] != capacity["kv_pages"] * capacity["page_size"]):
            raise ValueError("native worker was not at the specified prepared, empty-pool boundary")
        return asset

    def qualify(self, queries, recording, timeout=None):
        first = queries[0]
        if any(q != first for q in queries):
            raise ValueError("one initialization must answer the same initial resource state")
        workers = WorkerCapture.observed(recording)
        assets = [self.initialized(worker, first) for worker in workers]
        roles = assets[0]["environment"]["worker_roles"]
        count = first["domain"]["instances"]["null"]["tp"]
        if (set(map(int, roles)) != set(range(count)) or any(a["environment"]["worker_roles"] != roles for a in assets)
                or {w.rank for w in workers} != {r["representative"] for r in roles.values()}):
            raise ValueError("worker initialization must observe every representative and bind every logical rank")
        name = "workers/binding.json"
        product = assets[0] if len(assets) == 1 else {"participants": assets}
        return {name: product}, [{"status": "qualified", "asset": name} for _ in queries]


class LocalSchedulerBatch(LocalRule):
    name = scheduler_batch_rule
    compiler_gap = "missing_template_compiler"
    conditions = ("worker", "batch_operands")

    def calibration_cuts(self, job, sequence_capacity):
        batch = job["batch"]
        rows = batch["requests"]
        if batch["cut"] != "model" or batch["phase"] != "prefill" or len(rows) < 2:
            return {}
        sizes = [sum(n for _, n in row["prompt"]) for row in rows]
        length = max(sizes)
        longest = rows[sizes.index(length)]
        config = job["configuration"]
        page = config["page_size"]

        # The longest member may be admitted alone. Bound the measured endpoint by the original batch and native limits.
        room = sequence_capacity - longest["output_tokens"] if sequence_capacity is not None else length
        upper = min((length + page - 1) // page * page, sum(sizes), room,
                    config["max_prefill_tokens"], config["max_total_tokens"])
        prompt = longest["prompt"].copy()
        token, count = prompt[-1]
        prompt[-1] = [token, count + max(0, upper - length)]
        result = {"single-prefill": batch | {"requests": [longest | {"prompt": prompt}]}}

        # Two live cohorts can join on Decode. One step of progress distinguishes the cohorts without replaying long
        # requests.
        active = [row for row in rows if row["output_tokens"] > 1]
        if len(active) < 2:
            return result
        left, right = active[:-1], active[-1]
        lead = int(all(row["output_tokens"] > 2 for row in left))
        priming = [{"admit": [r["id"] for r in left], "decode_steps": lead},
                   {"admit": [right["id"]], "decode_steps": 0}]
        final = [r | {"output_tokens": 2 + lead} for r in left] + [right | {"output_tokens": 2}]
        joined = batch | {"phase": "decode", "requests": final, "priming": priming}
        return result | {"joined-forward": joined, "joined-result": joined | {"cut": "host"}}

    def calibration_products(self, recording, rank, model):
        batches = recording.inputs["job"]["calibration"]
        if batches.keys() != recording.manifest["calibration"].keys():
            raise ValueError("native calibration did not return each planned batch cut")
        products = {}
        for name, batch in batches.items():
            worker, timed = (WorkerCapture(recording, rank, part, cut=name) for part in ("structure", "timing"))
            facts, catalog = worker.measure(model, recording.directory.parent)
            boundary = worker.read("boundary")
            BatchReadback.observed_asset(batch, boundary, facts)
            for row, member in zip(batch["requests"], boundary["members"], strict=True):
                length = sum(n for _, n in row["prompt"])
                tokens = (0, length) if batch["phase"] == "prefill" else (length + row["output_tokens"] - 2, 1)
                if (member["id"], member["context_tokens"], member["new_tokens"]) != (row["id"], *tokens):
                    raise ValueError("native calibration selected different members or token positions")
            host = HostCostSamples.local(worker, timed, catalog)
            device = CostSamples.worker_table(facts, catalog)

            # Copies in serving may wait for an input. The ready-buffer sampler owns their durations.
            if copies := [[d, i, f] for work, d, i, f in catalog if "copy" in work]:
                device["required_work"] = copies
            for kind, table, parts in (("host", host, ["structure", "timing"]), ("device", device, ["structure"])):
                probe = {"cut": name, "rank": rank, "parts": parts}
                products[f"samples/{name}-{kind}.json"] = table | {"native_probe": probe}
            if batch["cut"] == "host":
                merges = [scope for scope in facts["scopes"] if scope["role"] == "batch_transform"
                          and scope["work"]["context"]["operation"] == "merge"]
                if len(merges) != 1 or not all(effect["finished"] for effect in boundary["effects"]):
                    raise ValueError("joined cohorts must reach one native merge and release their request slots")
                merge = merges[0]
                left, right = (cohort["admit"] for cohort in batch["priming"])
                groups = [[row["id"] for row in merge["work"]["context"][key]] for key in ("left", "right", "result")]
                if groups != [left, right, left + right]:
                    raise ValueError("native merge changed the planned cohort membership or order")
                products[f"structure/{name}.json"] = worker.read("facts") | {"export_site": merge["site"]}
        return products

    def matches(self, query):
        model_guard = (query.get("action") == "run_batch" and query.get("status") == "unmatched_conditions"
                       and query.get("cut", {}).get("cause") == "guard")
        return (missing_root(query) or model_guard) and self.supports(query)

    @staticmethod
    def supports(query):
        if BatchOperands.cut(query) is None or not WorkerDomain.dense_worker(query["domain"]):
            return False
        impl = query["domain"]["implementation"]
        return impl["cache"]["disabled"] and not any(impl["scheduling"].values())

    @staticmethod
    def session_key(query, deployment):
        cut = BatchOperands.cut(query)
        queues = BatchOperands.queue_input(query) if cut and cut[0] == "scheduler" else None
        separate = query.get("context", {}).get("right", []) if cut and cut[0] == "selection" else []
        if queues is not None:
            running = {m["id"] for m in queues["running"]}
            separate = [m for m in queues["previous"] if m["id"] not in running]
        return query["domain"], cut, separate, query.get("native_inputs"), query["action"] == "run_batch", queues

    @staticmethod
    def priming(phase, members, separate, rows):
        arrivals = defaultdict(list)
        right = {member["id"] for member in separate}
        for member, row in zip(members, rows, strict=True):
            if phase in {"admission", "selection", "scheduler"}:
                produced = member["generated_tokens"]
                if phase == "admission" and (produced or member["committed_tokens"]):
                    raise EvidenceGap("unbound_batch_progress", required="cold_waiting_queue")
            else:
                prompt = sum(count for _, count in row["prompt"])
                produced = 0 if phase == "prefill" else member["context_tokens"] - prompt + 1
                if (member["phase"] != phase or (phase == "decode" and (member["new_tokens"] != 1 or produced < 1))
                        or (phase == "prefill" and (member["context_tokens"], member["new_tokens"]) != (0, prompt))):
                    raise EvidenceGap("unbound_batch_progress", required="native_prompt_and_context")
            if not 0 <= produced <= row["output_tokens"] or phase == "decode" and produced == row["output_tokens"]:
                raise EvidenceGap("unbound_batch_progress", required="reachable_local_generation")
            if produced:
                arrivals[produced, member["id"] in right].append(member["id"])

        # Prefill advances only its entrants. Decode advances all active members in this non-mixed scheduler.
        cohorts = sorted(arrivals, key=lambda entry: (-entry[0], entry[1]))
        return [{"admit": arrivals[current], "decode_steps": current[0] - following[0]}
                for current, following in pairwise([*cohorts, (1, False)])]

    @classmethod
    def job(cls, session, deployment):
        domain, cut, separate, inputs, model, queues = session
        if inputs is None:
            raise EvidenceGap("unbound_scheduler_inputs", required=["configuration", "requests"])
        phase, members = cut
        identities = [row["id"] for row in inputs["requests"]]
        if len(set(identities)) != len(identities) or identities != [member["id"] for member in members]:
            raise ValueError("local inputs must name the requested members once in their original order")
        prefix = cls.priming(phase, members, separate, inputs["requests"])
        configuration = NativeConfiguration(inputs["configuration"])
        requested = WorkerDomain.observed([{"domain": domain}])
        if configuration.structure() != requested["instances"]["null"]:
            raise ValueError("supplied native configuration differs from the requested worker")
        batch = {"cut": "model" if model else "host", "phase": phase, "requests": inputs["requests"], "priming": prefix}
        job = {"configuration": configuration.values, "model": requested["model"], "batch": batch}
        if queues is not None:
            batch["waiting"] = [m["id"] for m in queues["waiting"]]
        return LocalCut.workers(job, deployment, configuration.tp, domain=domain)

    @classmethod
    def work_job(cls, queries, deployment):
        actions = [q.get("invocation", {}) | {"domain": q["domain"]} for q in queries]
        if not all("native_inputs" in action and cls.supports(action) for action in actions):
            raise EvidenceGap("unbound_native_work", required="supported_native_invocation_and_request_operands")
        return cls.job(cls.common_session(actions, deployment), deployment)

    def cut(self, queries, session, deployment, arguments):
        self.binary()
        cut = self.job(session, deployment)
        calibration = self.calibration_cuts(cut.job, queries[0]["parameters"].get("sequence_capacity"))
        if calibration:
            cut.job["calibration"] = calibration
            cut.host_timing = True
        return cut

    def qualify(self, queries, recording, timeout=None):
        worker = WorkerCapture(recording, WorkerDomain.rank(queries[0]["domain"]))
        facts, bounds = worker.read("facts"), worker.read("boundary")
        actual = WorkerDomain.execution_domain(facts["environment"])
        batch = recording.inputs["job"]["batch"]
        if any(actual != q["domain"] for q in queries):
            raise ValueError("native batch acquisition ran on a different worker")
        asset = BatchReadback.observed_asset(batch, bounds, facts)
        if (asset == "structure/model.json") != (queries[0]["action"] == "run_batch"):
            raise ValueError("observed native batch work differs from the requested action")
        if batch["phase"] == "selection":
            observed = [scope["work"]["context"] for scope in facts["scopes"] if scope["role"] == "batch_transform"]
            if any(query["context"] not in observed for query in queries):
                raise EvidenceGap("unobserved_batch_transition", observed=observed)
        elif batch["phase"] == "scheduler":
            if any(bounds["queues"] != BatchOperands.queue_input(query) for query in queries):
                raise EvidenceGap("unobserved_scheduler_input", observed=bounds["queues"])
        elif batch["phase"] != "admission":
            observed = BatchReadback.geometry(bounds["members"])
            if any(observed != BatchReadback.geometry(query["members"]) for query in queries):
                raise ValueError("native batch readback differs from requested membership or token geometry")
        model = self.compiler(timeout)
        products, answers = self.select(queries, worker, facts, bounds["phase"], asset, model)
        if products and "calibration" in recording.inputs["job"]:
            support = self.calibration_products(recording, worker.rank, model)
            for answer in answers:
                if answer["status"] == "qualified":
                    answer["support"] = sorted(support)
            products.update(support)
        return products, answers

    def select(self, queries, worker, facts, phase, asset, modeling):

        # Reuse production selection, including guards and ambiguity. Only a directly observed input can fill a gap.
        answers = []
        with TemporaryDirectory(dir=worker.recording.directory.parent) as scratch:
            root = Path(scratch)
            graph, library = root / "graph.json", root / "templates.json"
            job = worker.job
            if asset == "structure/model.json" and phase == "decode" and job["configuration"]["tp"] == 1:
                evidence, catalog = worker.catalog(modeling, root)
                runner = {key: job[key] for key in ("model", "configuration")}
                sampling = AttentionCatalog.from_catalog(worker.source, worker.source / "sdk/trace.json", worker.rank,
                                                        catalog, evidence, runner, device_runtime().scoped_tasks.source)
                if sampling["cuts"]:
                    facts = facts | {"sampling": sampling}
            else:
                modeling.run_model("project", graph, asset_path(worker.source, worker.manifest["facts"]))
            modeling.run_model("templates", library, graph)
            for i, query in enumerate(queries):
                demand, answer = root / "query.json", root / f"answer-{i}.json"
                write_json(demand, query)
                modeling.run_model("query-template", answer, library, demand)
                selection = read_json(answer)
                covered = selection["status"] == "covered" and "observation" in selection
                answers.append({"status": "qualified", "asset": asset} if covered
                               else {"status": "untriggered", "selection": selection})
        return LocalRule.asset(asset, facts, answers)


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
            operands[caller, dtype, width][size].add(WorkerDomain.rank(query["domain"]))
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
            row = rows.get(key)
            if (domain["implementation"]["framework_revision"] != inputs["framework_revision"]
                    or self.runtime(domain) != self.runtime(recording.manifest["runtime"])):
                answers.append({"status": "untriggered", "missing": "target_runtime"})
            elif (not row or planned[key][0] != width or WorkerDomain.rank(domain) not in planned[key][1]
                    or not row["verified_operand"] or not row["observations_ns"]):
                answers.append({"status": "untriggered", "missing": "matching_synchronous_copy_and_destination"})
            else:
                name, _, keys = groups.setdefault(cost_identity(domain), (f"samples/{len(groups)}.json", domain, set()))
                keys.add(key)
                answers.append({"status": "qualified", "asset": name, "observations": len(row["observations_ns"])})

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


class LocalAttention(LocalRule):
    name = RuleName("decode_attention")
    worker = "--local-attention"
    runtime_trace = False
    conditions = ("worker", "attention_cut")

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query["definition"]

    def __init__(self, sampling_sources):
        self.sampling_sources = sampling_sources

    @staticmethod
    def matches(query):
        if query.get("work") != "device":
            return False
        domain, features, definition = query["domain"], query["features"], query["definition"]
        return (features["phase"] == "decode" and "transport" not in features and WorkerDomain.dense_worker(domain)
                and domain["parallel"]["logical_tp"] == 1
                and definition.get("role") == "attention" and not definition["children"])

    def cut(self, queries, session, deployment, arguments):
        domain, definition = session
        contexts = set()
        for q in queries:
            lengths = LocalAttention.contexts(q["features"]["axes"])
            if not lengths:
                raise EvidenceGap("unbound_decode_context", features=q["features"])
            contexts.add(lengths)

        selected = None
        for recording, source in self.sampling_sources() if self.sampling_sources else ():
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
        platform = platform_backend().deployment(deployment, 1, False)
        revision = domain["implementation"]["framework_revision"]
        job = {"runner": runner, "site": cut["site"], "contexts": [list(batch) for batch in sorted(contexts)]}
        return LocalCut(job, platform.devices, revision, runner["model"]["path"],
                        {"template_recording": recording, "cut": cut, "domain": domain})

    def committed(self, call, lengths):
        if "kv" not in call:
            return False
        prefill, decode = call["kv"]["prefill"], call["kv"]["decode"]
        return (all(row["matching_operands"] and row["preserved_prefix"] for row in (prefill, decode))
                and prefill["slots"] == decode["slots"] and prefill["committed"] == list(lengths)
                and decode["committed"] == [n + 1 for n in lengths]
                and prefill["locations"] == [row[:-1] for row in decode["locations"]]
                and prefill["written"] == [loc for row in prefill["locations"] for loc in row]
                and decode["written"] == [row[-1] for row in decode["locations"]])

    def qualify(self, queries, recording, timeout=None):
        domain, cut = recording.inputs["domain"], recording.inputs["cut"]
        record = self.recording_part(recording, "measurements")
        native = domain["implementation"]
        same_worker = (record["rank"] == WorkerDomain.rank(domain) and record["model"] == native["model"]
                       and record["load_format"] == native["load_format"]
                       and CopyOperand.runtime(record["runtime"]) == CopyOperand.runtime(domain))
        calls = defaultdict(list)
        for call in record["calls"]:
            if call["site"] == cut["site"]:
                calls[tuple(call["context"])].append(call)

        definition, name = cut["definition"], "samples/0.json"
        samples = ServiceSamples(domain, [definition])
        measured, published, answers = {}, set(), []
        for q in queries:
            lengths, interval = self.contexts(q["features"]["axes"]), q["interval"]
            if lengths not in measured:
                group = calls[lengths]
                measured[lengths] = self.measured(domain, definition, cut["parts"], group, lengths) if (
                    same_worker and group and all(self.committed(call, lengths) for call in group)
                ) else (None, "unobserved_worker_or_kv")
            costs, detail = measured[lengths]
            if (costs is None or interval not in costs or q["domain"] != domain
                    or q["definition"] != definition or not self.matches(q)):
                answers.append({"status": "untriggered", "detail": detail})
                continue

            # One measured interval may answer several original queries.
            if (lengths, interval) not in published:
                samples.add(0, interval, q["features"], costs[interval])
                published.add((lengths, interval))
            answers.append({"status": "qualified", "asset": name, "observations": 1})
        detail = [{"contexts": lengths, "result": result} for lengths, (_, result) in measured.items()]
        return self.asset(name, samples.document({"site": cut["site"], "measurements": detail}), answers)

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

    def same_heads(self, call, definition):
        heads = call["heads"]
        widths = {"argument.q": heads["query"] * heads["key_dim"], "argument.k": heads["key"] * heads["key_dim"],
                  "argument.v": heads["value"] * heads["value_dim"]}
        seen = set()
        for port in definition["ports"]:
            name = port["name"]
            if name not in widths or port["direction"] != "input":
                continue
            tensor, shape = port["tensor"], port["tensor"]["shape"]
            if (not shape or shape[0] != "model_tokens" or not all(type(x) is int for x in shape[1:])
                    or prod(shape[1:]) != widths[name] or call["operands"].get(name) !=
                    {"shape": [call["query_rows"], *shape[1:]], "dtype": tensor["dtype"]}):
                return False
            seen.add(name)
        return seen == widths.keys()

    def measured(self, domain, definition, parts, calls, lengths):
        if not parts:
            return None, "source_has_no_single_stream_partition"
        matching = [c for c in calls if c["query_rows"] == len(lengths)]
        if len(matching) != 2 or "profile" in matching[0] or "profile" not in matching[1]:
            return None, "native_batch_geometry_not_repeated"
        plain, measured = matching
        native = domain["implementation"]
        for call in matching:
            if (call["kv_layout"] != domain["kv_layout"] or call["backend"] != native["attention_backend"]
                    or call["implementation"] != definition["implementation"]
                    or call["page_size"] != native["page_size"] or not call["full_context"]
                    or not self.same_heads(call, definition)):
                return None, "attention_domain_or_read_extent_changed"
        if "reason" in measured:
            return None, measured["reason"]
        rows = measured["tasks"]
        if [tuple(task[:2]) for task in rows] != [tuple(task) for _, tasks in parts for task in tasks]:
            return None, "device_task_sequence_changed"
        durations = iter(row[2] for row in rows)
        costs = {interval: sum(next(durations) for _ in tasks) for interval, tasks in parts}
        return costs, {"events_only_ns": plain["envelope_ns"], "with_profiler_ns": measured["envelope_ns"]}
