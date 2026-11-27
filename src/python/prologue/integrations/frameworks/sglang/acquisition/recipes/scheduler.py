"""Observe worker initialization and scheduler-selected native batches."""

from collections import defaultdict
from itertools import groupby, pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import read_json, write_json
from prologue.host.acquisition.rules import EvidenceGap, missing_root
from prologue.host.libraries.release import asset_path
from prologue.integrations.bindings import RuleName, device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.shared.batches import BatchOperands
from prologue.integrations.frameworks.sglang.shared.configuration import AttentionCatalog, NativeConfiguration
from prologue.integrations.frameworks.sglang.shared.resources import CacheInputs

from ..contracts import LocalCut, LocalRule, WorkerDomain, scheduler_batch_rule
from ..evidence import CostSamples
from ..host_samples import HostCostSamples
from ..readback import BatchReadback, WorkerCapture


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
        config = job["configuration"]
        if (batch["cut"] != "model" or batch["phase"] != "prefill" or len(rows) < 2
                or config["prefill_chunk_tokens"]):
            return {}

        sizes = [sum(n for _, n in row["prompt"]) for row in rows]
        length = max(sizes)
        longest = rows[sizes.index(length)]
        page = config["page_size"]

        # The longest member may be admitted alone. Bound the measured endpoint by the original batch and native limits.
        room = sequence_capacity - longest["output_tokens"] if sequence_capacity is not None else length
        upper = min((length + page - 1) // page * page, sum(sizes), room,
                    config["max_prefill_tokens"], config["max_total_tokens"])
        prompt = longest["prompt"].copy()
        token, count = prompt[-1]
        prompt[-1] = [token, count + max(0, upper - length)]

        # Calibration starts with its own empty pool. Target cache seeds and queue state cannot carry over.
        calibration = {"cut": "model", "phase": "prefill", "priming": []}
        if "rank" in batch:
            calibration["rank"] = batch["rank"]
        result = {"single-prefill": calibration | {"requests": [longest | {"prompt": prompt}]}}

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
        joined = calibration | {"phase": "decode", "requests": final, "priming": priming}

        return result | {"joined-forward": joined, "joined-result": joined | {"cut": "host"}}

    def calibration_products(self, recording, rank, model):
        batches = recording.inputs["job"]["calibration"]
        if batches.keys() != recording.manifest["calibration"].keys():
            raise ValueError("native calibration did not return each planned batch cut")

        products, rejected = {}, {}
        for name, batch in batches.items():
            worker, timed = (WorkerCapture(recording, rank, part, cut=name) for part in ("structure", "timing"))
            boundary = worker.read("boundary")
            expected = []
            for row in batch["requests"]:
                length = sum(n for _, n in row["prompt"])
                context, count = (0, length) if batch["phase"] == "prefill" else (length + row["output_tokens"] - 2, 1)
                expected.append({
                    "id": row["id"], "phase": batch["phase"], "context_tokens": context, "new_tokens": count,
                })

            observed = BatchReadback.geometry(boundary["members"])
            if observed != expected:
                rejected[name] = {
                    "status": "unobserved_calibration_geometry", "expected": expected, "observed": observed,
                }
                continue

            facts, catalog = worker.measure(model, recording.directory.parent)
            BatchReadback.observed_asset(batch, boundary, facts)
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

        return products, rejected

    def matches(self, query):
        model_guard = (query.get("action") == "run_batch" and query.get("status") == "unmatched_conditions"
                       and query.get("cut", {}).get("cause") == "guard")
        return (missing_root(query) or model_guard) and self.supports(query)

    @staticmethod
    def supports(query):
        cut = BatchOperands.cut(query)
        if cut is None or not WorkerDomain.dense_worker(query["domain"]):
            return False

        # A radix lookup needs reproducible trie operands. Other cuts qualify against their observed native path.
        return (query["action"] != "cache_match" or query["domain"]["implementation"]["cache"]["disabled"]
                or LocalSchedulerBatch.cache_match(query) is not None)

    @staticmethod
    def cache_match(query):
        if query["action"] != "cache_match" or (cut := BatchOperands.cut(query)) is None:
            return None

        request = query.get("context", {}).get("request")
        if request is None or all(member["id"] != request["id"] for member in cut[1]):
            return None

        expected = ("query_tokens", "matched_tokens", "split_prefix_tokens", "split_suffix_tokens")
        parameters = query["parameters"]
        if parameters.keys() >= set(expected):
            return {"request": request["id"], **{key: parameters[key] for key in expected}}

    @staticmethod
    def cache_pressure(query):
        condition = query.get("condition", {})
        if query["action"] != "segment" or condition.get("owner") != "eviction":
            return None

        observed = query.get("continuation", {})
        if observed.get("exit") == "pages.free" and "eviction_tokens" in query["parameters"]:
            return query["parameters"]["eviction_tokens"]

    @staticmethod
    def session_key(query, deployment):
        cut = BatchOperands.cut(query)
        phase = cut[0] if cut else None
        queues = query.get("preparation") if phase in BatchOperands.replay_phases else None
        if queues is None and phase in {"scheduler", "selection", "mixed"}:
            queues = BatchOperands.queue_input(query)

        separate = query.get("context", {}).get("right", []) if phase == "selection" else []
        if phase == "scheduler":
            running = {m["id"] for m in queues["running"]}
            separate = [m for m in queues["previous"] if m["id"] not in running]

        # Bookkeeping inserts its own entry before matching. Only external lookups need a seeded trie.
        owners = {caller["owner"] for caller in query.get("callers", ())}
        cache_match = None
        if queues is not None and owners.isdisjoint(CacheInputs.operations):
            cache_match = LocalSchedulerBatch.cache_match(query)

        return (query["domain"], cut, separate, query.get("native_inputs"), query["action"] == "run_batch", queues,
                cache_match, LocalSchedulerBatch.cache_pressure(query))

    @staticmethod
    def mixed_frontier(arrivals, chunks, config):
        decoders = sum(map(len, arrivals.values()))
        width = config["prefill_chunk_tokens"] - decoders
        context, = chunks.values()
        if width <= 0 or context % width:
            raise EvidenceGap("unbound_mixed_frontier", expected_full_chunk=width, committed=context)

        # The continuing request's earlier chunks each advanced every active Decode member.
        earlier = context // width
        if any(produced <= earlier for produced, _ in arrivals):
            raise EvidenceGap("unbound_mixed_frontier", required_decode_steps=earlier)
        return {(produced - earlier, group): ids for (produced, group), ids in arrivals.items()}

    @staticmethod
    def previous_prefill(queues, rows):
        previous = queues["previous"]
        if queues["running"] or len(previous) < 2:
            return None

        lengths = {row["id"]: sum(length for _, length in row["prompt"]) for row in rows}
        continuing, fresh = {}, {}
        for member in previous:
            identity, extent = member["id"], member["extend"]
            if identity not in lengths or extent is None:
                return None
            begin, end = extent
            if (not 0 <= begin < end <= lengths[identity] or end != member["committed_tokens"]
                    or member["generated_tokens"] != int(end == lengths[identity])):
                return None

            if begin:
                continuing[identity] = begin
            else:
                fresh[identity] = end

        # A continuing Prefill can finish in the same native batch that admits a new request.
        if len(continuing) > 1:
            return None
        return [{"admit": list(group), "prefill_until": group, "decode_steps": 0}
                for group in (continuing, fresh) if group]

    @staticmethod
    def priming(phase, members, separate, rows, chunked, config, cached):
        arrivals = defaultdict(list)
        chunks = {member["id"]: member["committed_tokens"] for member in chunked}
        right = {member["id"] for member in separate}
        for member, row in zip(members, rows, strict=True):
            if phase in {"admission", "selection", "scheduler"}:
                produced = member["generated_tokens"]
                if phase == "admission" and (produced or member["committed_tokens"]):
                    raise EvidenceGap("unbound_batch_progress", required="cold_waiting_queue")
                if row["id"] in chunks and (produced or not 0 < chunks[row["id"]] < sum(n for _, n in row["prompt"])):
                    raise EvidenceGap("unbound_chunk_progress", member=row["id"])
            else:
                prompt = sum(count for _, count in row["prompt"])
                context, new = member["context_tokens"], member["new_tokens"]
                decoding = member["phase"] == "decode"
                produced = context - prompt + 1 if decoding else 0
                if (decoding and (new != 1 or produced < 1)
                        or not decoding and not 0 <= context < context + new <= prompt):
                    raise EvidenceGap("unbound_batch_progress", required="native_prompt_and_context")

                if not decoding and context and row["id"] not in cached:
                    chunks[row["id"]] = context

            if (not 0 <= produced <= row["output_tokens"]
                    or phase in {"decode", "mixed"} and produced == row["output_tokens"]):
                raise EvidenceGap("unbound_batch_progress", required="reachable_local_generation")
            if produced:
                arrivals[produced, member["id"] in right].append(member["id"])

        if len(chunks) > 1:
            raise EvidenceGap("unbound_chunk_frontier", members=sorted(chunks))
        if chunks and phase == "mixed":
            arrivals = LocalSchedulerBatch.mixed_frontier(arrivals, chunks, config)

        cohorts = sorted(arrivals, key=lambda entry: (-entry[0], entry[1]))
        planned = [{"admit": arrivals[current], "decode_steps": current[0] - following[0]}
                   for current, following in pairwise([*cohorts, (1, False)])]
        if chunks:
            planned.append({"admit": list(chunks), "prefill_until": chunks, "decode_steps": 0})
        return planned

    @staticmethod
    def cache_seeds(rows, match, page, vocab, *, primed=()):
        if not match["matched_tokens"]:
            return []

        requests = {row["id"]: row for row in rows}
        row = requests[match["request"]]
        matched, left, right = (match[key] for key in ("matched_tokens", "split_prefix_tokens", "split_suffix_tokens"))
        tokens = [token for token, length in row["prompt"] for _ in range(length)]
        if not (0 <= left <= matched <= match["query_tokens"] < len(tokens)
                and all(value % page == 0 for value in (matched, left, right)) and bool(left) == bool(right)):
            raise EvidenceGap("unreachable_prefix_partition", operands=match)

        # Replaying the original Prefill can create the branch. Seeding it early changes that Prefill's own hit.
        parent = matched - left
        next_page = slice(matched, matched + page)
        query_continues = matched + page <= match["query_tokens"]
        for member in primed:
            if member["extend"] is None:
                continue
            begin, end = member["extend"]
            if end // page * page != matched + right or (left and begin != parent):
                continue

            prompt = requests[member["id"]]["prompt"]
            observed = [token for token, length in prompt for _ in range(length)]
            if (observed[:matched] == tokens[:matched]
                    and not (right and query_continues and observed[next_page] == tokens[next_page])):
                return []

        # Insert an optional parent and an existing branch whose suffix differs beyond the desired match.
        paths = [tokens[:parent]] if left and parent else []
        paths.append(tokens[:matched] + [(tokens[matched] + 1) % vocab] * right)
        return [{"id": f"{row['id']}/cache-seed/{index}",
                 "prompt": [[token, sum(1 for _ in run)] for token, run in groupby(path)],
                 "next_token": row["next_token"], "output_tokens": 1} for index, path in enumerate(paths)]

    @staticmethod
    def pressure_seeds(members, rows, config, needed, vocab):
        page = config["page_size"]
        committed = sum(((m["context_tokens"] + page - 1) // page) * page for m in members)
        free = ((needed - 1) // page) * page
        parked = config["max_total_tokens"] - committed - free
        if not page <= parked < config["max_total_tokens"] or parked % page:
            raise EvidenceGap("unreachable_eviction_pressure", cached_tokens=parked, required_tokens=needed)

        # Reserve only the pages needed to put the next native allocation below its requested capacity.
        active = {row["prompt"][0][0] for row in rows}
        token = next(candidate for candidate in range(vocab) if candidate not in active)
        return [{"id": f"{rows[0]['id']}/pressure-seed", "prompt": [[token, parked]],
                 "next_token": token, "output_tokens": 1}]

    @classmethod
    def model_prefixes(cls, members, rows, page, vocab):
        by_id = {row["id"]: row for row in rows}
        cached = {}
        for member in members:
            context, identity = member["context_tokens"], member["id"]
            if member["phase"] != "prefill" or not context or context % page:
                continue

            # A model cut consumes KV values for the same input tokens; the native cache can prepare whole pages.
            prompt = sum(length for _, length in by_id[identity]["prompt"])
            match = {"request": identity, "matched_tokens": context, "query_tokens": prompt - 1,
                     "split_prefix_tokens": 0, "split_suffix_tokens": 0}
            cached[identity] = cls.cache_seeds(rows, match, page, vocab)
        return cached

    @classmethod
    def job(cls, session, deployment):
        domain, cut, separate, inputs, model, queues, cache_match, pressure = session
        if inputs is None:
            raise EvidenceGap("unbound_scheduler_inputs", required=["configuration", "requests"])

        phase, members = cut
        identities = [row["id"] for row in inputs["requests"]]
        if len(set(identities)) != len(identities) or identities != [member["id"] for member in members]:
            raise ValueError("local inputs must name the requested members once in their original order")

        configuration = NativeConfiguration(inputs["configuration"])
        values, rows = configuration.values, inputs["requests"]
        vocab = domain["implementation"]["model"]["vocab_size"]
        replay = queues.get("replay") if queues is not None else None
        prefix = cls.previous_prefill(queues, rows) if queues is not None and replay is None else None
        cached = {}
        if prefix is None and replay is None:
            if model and values["prefix_cache"]:
                cached = cls.model_prefixes(members, rows, values["page_size"], vocab)
            chunked = queues["chunked"] if phase == "scheduler" else ()
            prefix = cls.priming(phase, members, separate, rows, chunked, values, cached)

        requested = WorkerDomain.observed([{"domain": domain}])
        if configuration.structure() != requested["instances"]["null"]:
            raise ValueError("supplied native configuration differs from the requested worker")

        batch = {"cut": "model" if model else "host", "phase": phase, "requests": rows, "priming": prefix or []}
        if replay is not None:
            batch["replay"] = {**replay, "prior_requests": inputs["prior_requests"]}
        job = {"configuration": values, "model": requested["model"], "batch": batch}
        if phase == "scheduler":
            batch["waiting"] = [m["id"] for m in queues["waiting"]]
        if cache_match is not None and replay is None:
            admitted = {identity for cohort in prefix for identity in cohort["admit"]}
            primed = (member for member in members if member["id"] in admitted)
            seeds = cls.cache_seeds(rows, cache_match, values["page_size"], vocab, primed=primed)
            if seeds:
                batch["cache_seeds"] = seeds
        if pressure is not None:
            batch["cache_pressure"] = pressure
            if replay is None:
                batch["cache_seeds"] = cls.pressure_seeds(members, rows, values, pressure, vocab)
        if cached:
            batch["cache_seeds"] = [seed for seeds in cached.values() for seed in seeds]

        return LocalCut.workers(job, deployment, configuration.tp, domain=domain)

    @classmethod
    def work_job(cls, queries, deployment):
        actions = cls.invocations(queries)
        if not all("native_inputs" in action and cls.supports(action) for action in actions):
            raise EvidenceGap("unbound_native_work", required="supported_native_invocation_and_request_operands")
        return cls.job(cls.common_session(actions, deployment), deployment)

    @staticmethod
    def invocations(queries):
        return [q.get("invocation", {}) | {"domain": q["domain"]} for q in queries]

    def cut(self, queries, session, deployment, arguments):
        self.binary()
        cut = self.job(session, deployment)
        batch = cut.job["batch"]
        if "rank" in batch and not cut.job["single_device"]:
            roles = queries[0]["domain"]["worker_roles"].values()
            batch["peer_ranks"] = sorted({batch["rank"], *(role["representative"] for role in roles)})

        calibration = self.calibration_cuts(cut.job, queries[0]["parameters"].get("sequence_capacity"))
        if calibration:
            cut.job["calibration"] = calibration
            cut.host_timing = True
        return cut

    @classmethod
    def readback(cls, queries, worker):
        facts, bounds = worker.read("facts"), worker.read("boundary")
        actual = WorkerDomain.execution_domain(facts["environment"])
        batch = worker.job["batch"]

        if any(actual != q["domain"] for q in queries):
            raise ValueError("native batch acquisition ran on a different worker")
        asset = BatchReadback.observed_asset(batch, bounds, facts)
        if (asset == "structure/model.json") != (queries[0]["action"] == "run_batch"):
            raise ValueError("observed native batch work differs from the requested action")

        if batch["phase"] == "scheduler":
            if any(bounds["queues"] != BatchOperands.queue_input(query) for query in queries):
                raise EvidenceGap("unobserved_scheduler_input", observed=bounds["queues"])
        elif batch["phase"] not in {"admission", "selection"}:
            observed = BatchReadback.geometry(bounds["members"])
            for query in queries:
                expected = BatchReadback.geometry(BatchOperands.cut(query)[1])
                if observed != expected:
                    raise EvidenceGap("unobserved_batch_geometry", expected=expected, observed=observed)

        transitions = [query["context"] for query in queries if query["action"] == "batch_transform"]
        if transitions:
            observed = [scope["work"]["context"] for scope in facts["scopes"] if scope["role"] == "batch_transform"]
            if any(context not in observed for context in transitions):
                raise EvidenceGap("unobserved_batch_transition", observed=observed)

        for query in queries:
            if match := cls.cache_match(query):
                cls.check_cache_match(match, query, facts)

        if "cache_pressure" in batch:
            cls.check_cache_pressure(batch["cache_pressure"], facts)

        return facts, bounds["phase"], asset

    def qualify(self, queries, recording, timeout=None):
        worker = WorkerCapture(recording, WorkerDomain.rank(queries[0]["domain"]))
        facts, phase, asset = self.readback(queries, worker)
        model = self.compiler(timeout)
        products, answers = self.select(queries, worker, facts, phase, asset, model)
        if products:
            self.attach_support(products, answers, self.peer_products(queries, recording, worker.rank, model))

        if products and "calibration" in recording.inputs["job"]:
            support, rejected = self.calibration_products(recording, worker.rank, model)
            self.attach_support(products, answers, support)
            if rejected:
                for answer in answers:
                    if answer["status"] == "qualified":
                        answer["rejected_calibration"] = rejected

        return products, answers

    def peer_products(self, queries, recording, rank, model):
        products = {}
        origin = queries[0]["domain"]
        for worker in WorkerCapture.observed(recording):
            if worker.rank == rank:
                continue
            role = origin["worker_roles"][str(worker.rank)]
            domain = WorkerDomain.execution_domain(WorkerDomain.bind_worker(origin, role))
            requests = [query | {"domain": domain} for query in queries]

            # A companion contributes only a real observation of the same demand on its own worker.
            facts, phase, asset = self.readback(requests, worker)
            observed, _ = self.select(requests, worker, facts, phase, asset, model)
            for name, value in observed.items():
                products[f"{name.removesuffix('.json')}-rank-{worker.rank}.json"] = value

        return products

    @staticmethod
    def attach_support(products, answers, support):
        if support:
            products.update(support)
            for answer in answers:
                if answer["status"] == "qualified":
                    answer.setdefault("support", []).extend(sorted(support))

    @staticmethod
    def check_cache_match(expected, query, facts):
        request = query["context"]["request"]
        owners = [caller["owner"] for caller in query.get("callers", ())]
        parameters = {key: value for key, value in expected.items() if key != "request"}
        reads = []
        for scope in facts["scopes"]:
            work = scope["work"]
            if (scope["role"] == "cache_match" and work["context"].get("request") == request
                    and [caller["owner"] for caller in work.get("callers", ())] == owners):
                reads.append(scope["parameters"])

        if len(reads) != 1 or any(reads[0].get(key) != value for key, value in parameters.items()):
            raise EvidenceGap("unobserved_prefix_partition", expected=expected, observed=reads)

    @staticmethod
    def check_cache_pressure(expected, facts):
        for scope in facts["scopes"]:
            cut = scope["work"].get("continuation", {})
            if (cut.get("owner") == "eviction" and cut["exit"] == "pages.free"
                    and scope["parameters"].get("eviction_tokens") == expected):
                return
        raise EvidenceGap("unobserved_cache_eviction", required_tokens=expected)

    def select(self, queries, worker, facts, phase, asset, modeling):

        # Reuse production selection, including guards and ambiguity. Only a directly observed input can fill a gap.
        answers, catalog = [], None
        with TemporaryDirectory(dir=worker.recording.directory.parent) as scratch:
            root = Path(scratch)
            graph, library = root / "graph.json", root / "templates.json"
            job = worker.job

            if any(span["kind"] in {"device", "transfer"} for span in facts["spans"]):
                evidence, catalog = worker.catalog(modeling, root)
                if asset == "structure/model.json" and phase == "decode":
                    runner = {key: job[key] for key in ("model", "configuration")}
                    sampling = AttentionCatalog.from_catalog(
                        worker.source, worker.source / "sdk/trace.json", worker.rank, catalog, evidence, runner,
                        device_runtime().scoped_tasks.source,
                    )
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

        products, answers = LocalRule.asset(asset, facts, answers)
        if products and catalog is not None:
            self.attach_support(products, answers, CostSamples.device_support(evidence, catalog))

        return products, answers


rules = (LocalWorkerBinding, LocalSchedulerBatch)
