"""Qualify worker and batch queries against an initialized native scheduler."""

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.acquisition.rules import EvidenceGap, missing_root
from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.ascend.attention_cuts import AttentionCuts
from prologue.integrations.frameworks.sglang.ascend.batch_calibration import BatchCalibration
from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.ascend.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration
from prologue.integrations.frameworks.sglang.ingress import BatchOperands
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.deployment import AscendDeployment
from prologue.integrations.runtimes.docker import ModelingProcess


class LocalWorkerBinding:
    name = "sglang.ascend.worker_initialization"

    @staticmethod
    def session_key(query, deployment):
        return query

    @staticmethod
    def matches(query):
        if query.get("status") != "missing_worker_binding" or query.get("action") != "initialize_workers":
            return False
        structures = query["domain"]["instances"]
        if structures.keys() != {"null"} or len(query["instances"]) != 1:
            return False
        structure = structures["null"]
        return (structure["attention_backend"] == "ascend"
                and structure["pp"] == structure["ep"] == 1
                and "expert_layout" not in query["instances"][0]["readback"]["implementation"])

    @staticmethod
    def prepare(queries, deployment, arguments, limit):
        first = queries[0]
        if any(query != first for query in queries):
            raise EvidenceGap("different_sessions_required", conditions=["worker_initial_state"])
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
        placement = AscendDeployment(deployment, structure["tp"], False)
        if structure["tp"] > 1:
            job["single_device"] = placement.single_device
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, placement.devices, None, job,
                                                 "--local-worker", runtime_trace=True, model=model["path"])

    @staticmethod
    def initialized(worker, query):
        asset = worker.read("facts") | worker.read("binding")
        start = worker.read("boundary")
        if start != {"phase": "idle", "members": []} or not any(
            scope["bindings"].get("segment", {}).get("owner") == "selection" for scope in asset["scopes"]
        ):
            raise ValueError("initialized worker did not execute its idle selection boundary")
        env, capacity, initial = asset["environment"], asset["capacity"], asset["initial_state"]
        rank = worker.rank
        role = env["worker_roles"][str(rank)]
        sections = {"environment": env, "implementation": env["implementation"], "worker_roles": role,
                    "initial_state": initial}
        if (env["implementation"]["framework_revision"] != worker.recording.inputs["framework_revision"]
                or env["parallel"]["rank"] != rank or role["representative"] != rank or not role["modules"]
                or env["parallel"] != role["parallel"] or env["kv_layout"] != role["kv_layout"]):
            raise ValueError("native readback does not describe the initialized worker and its model")
        observed = WorkerDomain.observed([{"domain": env}])
        instance, = query["instances"]
        settings = instance["settings"]
        if observed != query["domain"] or any(set(fields) - sections[key].keys() for key, fields in instance["readback"].items()):
            raise ValueError("native worker does not satisfy the requested readback")
        if (capacity["page_size"] != settings["page_size"] or asset["configured_settings"] != settings
                or initial["logical_kv"] != "empty_after_native_flush" or initial["free_slots"] != capacity["slot_capacity"]
                or initial["free_tokens"] != capacity["kv_pages"] * capacity["page_size"]
                or initial["workspace"] != {"prefill_tokens": [1], "decode_steps": 1, "device_drained": True,
                                             "device_cache_emptied": False}):
            raise ValueError("native worker was not at the specified prepared, empty-pool boundary")
        return asset

    @classmethod
    def qualify(cls, queries, recording, timeout=None):
        first = queries[0]
        if any(q != first for q in queries):
            raise ValueError("one initialization must answer the same initial resource state")
        workers = WorkerCapture.observed(recording)
        assets = [cls.initialized(worker, first) for worker in workers]
        roles = assets[0]["environment"]["worker_roles"]
        count = first["domain"]["instances"]["null"]["tp"]
        if (set(map(int, roles)) != set(range(count)) or any(a["environment"]["worker_roles"] != roles for a in assets)
                or {w.rank for w in workers} != {r["representative"] for r in roles.values()}):
            raise ValueError("the initialized group must observe every native representative and bind every logical rank")
        name = "workers/binding.json"
        product = assets[0] if len(assets) == 1 else {"participants": assets}
        return {name: product}, [{"status": "qualified", "asset": name, "ranks": [w.rank for w in workers]} for _ in queries]


class LocalSchedulerBatch:
    name = "sglang.ascend.scheduler_batch"

    def __init__(self, arguments):
        self.arguments = arguments

    @classmethod
    def matches(cls, query):
        model_guard = (query.get("action") == "run_batch" and query.get("status") == "unmatched_conditions"
                       and query.get("cut", {}).get("cause") == "guard")
        return (missing_root(query) or model_guard) and cls.supports(query)

    @staticmethod
    def supports(query):
        if BatchOperands.cut(query) is None:
            return False
        domain = query["domain"]
        parallel, impl = domain["parallel"], domain["implementation"]
        return (domain["framework"] == "sglang" and "torch_npu" in domain and not domain["expert_layers"]
                and parallel["logical_pp"] == parallel["logical_ep"] == 1
                and impl["disaggregation_mode"] == "null" and impl["cache"]["disabled"]
                and not any(impl["scheduling"].values()))

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
        result = []
        for i, (count, other) in enumerate(cohorts):
            following = cohorts[i + 1][0] if i + 1 < len(cohorts) else 1
            result.append({"admit": arrivals[count, other], "decode_steps": count - following})
        return result

    @classmethod
    def job(cls, queries, deployment):
        first = queries[0]
        if "native_inputs" not in first:
            raise EvidenceGap("unbound_scheduler_inputs", required=["configuration", "requests"])
        session = cls.session_key(first, deployment)
        domain, cut, separate, inputs, model, queues = session
        phase, members = cut
        if any(cls.session_key(q, deployment) != session for q in queries):
            raise EvidenceGap("different_sessions_required", conditions=["worker", "batch_operands"])
        identities = [row["id"] for row in inputs["requests"]]
        if len(set(identities)) != len(identities) or identities != [member["id"] for member in members]:
            raise ValueError("local inputs must name the requested members once in their original order")
        prefix = cls.priming(phase, members, separate, inputs["requests"])
        configuration = NativeConfiguration(inputs["configuration"])
        requested = WorkerDomain.observed([{"domain": domain}])
        if configuration.structure() != requested["instances"]["null"]:
            raise ValueError("supplied native configuration differs from the requested worker")
        job = {"configuration": configuration.values, "model": requested["model"],
               "batch": {"cut": "model" if model else "host", "phase": phase, "requests": inputs["requests"], "priming": prefix}}
        if queues is not None:
            job["batch"]["waiting"] = [m["id"] for m in queues["waiting"]]
        placement = AscendDeployment(deployment, configuration.tp, False)
        if configuration.tp > 1:
            job["single_device"] = placement.single_device
            job["batch"]["rank"] = WorkerDomain.rank(domain)
        return job, placement.devices, domain["implementation"]["framework_revision"]

    @classmethod
    def work_job(cls, queries, deployment):
        actions = [q.get("invocation", {}) | {"domain": q["domain"]} for q in queries]
        if not all("native_inputs" in action and cls.supports(action) for action in actions):
            raise EvidenceGap("unbound_native_work", required="supported_native_invocation_and_request_operands")
        return cls.job(actions, deployment)

    def prepare(self, queries, deployment, arguments, limit):
        self.compiler_path()
        job, devices, revision = self.job(queries, deployment)
        calibration = BatchCalibration.cuts(job, queries[0]["parameters"].get("sequence_capacity"))
        if calibration:
            job["calibration"] = calibration
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, devices, revision, job,
                                                 "--local-worker", runtime_trace=True, host_timing=bool(calibration), model=job["model"]["path"])

    def qualify(self, queries, recording, timeout=None):
        worker = WorkerCapture(recording, WorkerDomain.rank(queries[0]["domain"]))
        facts, bounds = worker.read("facts"), worker.read("boundary")
        actual = WorkerDomain.execution_domain(facts["environment"])
        batch = recording.inputs["job"]["batch"]
        if any(actual != q["domain"] for q in queries):
            raise ValueError("native batch acquisition ran on a different worker")
        asset = self.observed_asset(batch, bounds, facts)
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
            observed = self.geometry(bounds["members"])
            if any(observed != self.geometry(query["members"]) for query in queries):
                raise ValueError("native batch readback differs from requested membership or token geometry")
        allowance = self.arguments.timeout if timeout is None else min(timeout, self.arguments.timeout)
        model = ModelingProcess(self.compiler_path(), self.arguments.modeling_image, allowance)
        products, answers = self.select(queries, worker, facts, bounds["phase"], asset, model)
        if products and "calibration" in recording.inputs["job"]:
            support = BatchCalibration.products(recording, worker.rank, model)
            for answer in answers:
                if answer["status"] == "qualified":
                    answer["support"] = sorted(support)
            products.update(support)
        return products, answers

    @classmethod
    def observed_asset(cls, batch, boundary, facts):
        if boundary["phase"] != batch["phase"]:
            raise ValueError("native batch acquisition ran in a different phase")
        if "output" in boundary:
            cls.check_model(boundary, facts)
            return "structure/model.json"
        if boundary["phase"] == "admission":
            if boundary["waiting"] != [row["id"] for row in batch["requests"]]:
                raise ValueError("native waiting queue differs from the supplied request order")
        elif boundary["phase"] not in {"selection", "scheduler"}:
            cls.check_host(batch, boundary, facts)
        return "structure/host.json"

    @staticmethod
    def geometry(members):
        return [{k: member[k] for k in ("id", "phase", "context_tokens", "new_tokens")} for member in members]

    @staticmethod
    def check_model(bounds, facts):
        native = bounds["output"]
        count = len(bounds["members"])
        if len(native["tokens"]) != count or native["tokens"] != native["relayed"] or native["device"] != "npu":
            raise ValueError("each sampled device token must reach its original native request slot")
        scopes = facts["scopes"]
        root, = (i for i, s in enumerate(scopes) if s["role"] == "run_batch" and s["parent"] == 0)

        def tensor(role, direction, name):
            ports = [p["tensor"] for i, s in enumerate(scopes) if (i == root or s["parent"] == root) and s["role"] == role
                     for p in s["ports"] if (p["direction"], p["name"]) == (direction, name)]
            if len(ports) != 1:
                raise ValueError(f"one native model cut requires one {role} {direction} port for {name}")
            return ports[0]

        logits = tensor("model", "output", "next_token_logits")
        tokens = tensor("run_batch", "output", "next_token_ids")
        if (tensor("sample", "input", "next_token_logits")["value"] != logits["value"]
                or tensor("sample", "output", "next_token_ids")["value"] != tokens["value"]):
            raise ValueError("model, sampler and batch must hand off the observed tensor values")
        if tokens["shape"] != [count] or tokens["dtype"] != native["dtype"]:
            raise ValueError("the observed model output port must describe the returned token tensor")

    @staticmethod
    def check_host(batch, bounds, facts):
        expected = []
        for row, member in zip(batch["requests"], bounds["members"], strict=True):
            prompt = sum(n for _, n in row["prompt"])
            count = 1 if batch["phase"] == "prefill" else member["context_tokens"] - prompt + 2
            expected.append({"id": row["id"], "generated_tokens": count, "finished": count == row["output_tokens"]})
        publication = [effect | {"tokens": [row["next_token"]]} for effect, row in zip(expected, batch["requests"], strict=True)]
        emitter = facts["environment"]["parallel"]["tp_rank"] == 0
        if bounds["effects"] != expected or bounds["publication"] != (publication if emitter else []):
            raise ValueError("native request state and receiving socket must agree on each produced token")
        scopes = facts["scopes"]
        bindings = [scope["bindings"] for scope in scopes]
        owners = {b.get("segment", {}).get("owner") for b in bindings}
        produced = [b["publication"] for b in bindings if "publication" in b]
        completed = [b["effects"] for b in bindings if "effects" in b and b["segment"]["owner"] == "result"]
        named = [{k: e[k] for k in ("id", "generated_tokens")} for e in expected]
        required = {"selection", "preparation", "result"} | ({"publication"} if emitter else set())
        if not required <= owners or produced != ([named] if emitter else []) or completed != [expected]:
            raise ValueError("observed host segments must account for the native request and publication effects")

        # The supplied token tensor is on the device. Its materialization must account for the actual batch payload.
        reads = [i for i, s in enumerate(scopes) if s["role"] == "host_read" and s["bindings"]["condition"]["input"] == "next_token_ids"]
        copies = [s for s in facts["spans"] if s["scope"] in reads and s.get("copy", {}).get("operand") == "next_token_ids"]
        if (len(reads) != 1 or len(copies) != 1 or copies[0]["copy"]["direction"] != "device_to_host"
                or copies[0]["operand_bytes"] != len(expected) * copies[0]["copy"]["element_bytes"]
                or not any(p["name"] == "next_token_ids" and p["direction"] == "input" for p in scopes[reads[0]]["ports"])):
            raise ValueError("the host read must consume the supplied device output for each batch member")

    @staticmethod
    def select(queries, worker, facts, phase, asset, modeling):

        # Reuse production selection, including guards and ambiguity. Only a directly observed input can fill a gap.
        answers = []
        with TemporaryDirectory(dir=worker.recording.directory.parent) as scratch:
            root = Path(scratch)
            graph, library = root / "graph.json", root / "templates.json"
            job = worker.recording.inputs["job"]
            if asset == "structure/model.json" and phase == "decode" and job["configuration"]["tp"] == 1:
                evidence, catalog = worker.catalog(modeling, root)
                runner = {key: job[key] for key in ("model", "configuration")}
                sampling = AttentionCuts.from_catalog(worker.source, worker.source / "sdk/trace.json",
                                                      worker.rank, catalog, evidence, runner)
                if sampling["cuts"]:
                    facts = facts | {"sampling": sampling}
            else:
                modeling.run_model("project", graph, worker.source / worker.manifest["facts"])
            modeling.run_model("templates", library, graph)
            for i, query in enumerate(queries):
                demand, answer = root / "query.json", root / f"answer-{i}.json"
                write_json(demand, query)
                modeling.run_model("query-template", answer, library, demand)
                selection = json.loads(answer.read_bytes())
                if selection["status"] == "covered" and "observation" in selection:
                    answers.append({"status": "qualified", "asset": asset, "phase": phase,
                                    "selection": selection})
                else:
                    answers.append({"status": "untriggered", "selection": selection})
        return ({asset: facts} if any(a["status"] == "qualified" for a in answers) else {}), answers

    def compiler_path(self):
        if self.arguments.binary is None:
            raise EvidenceGap("missing_template_compiler", required="--binary")
        return self.arguments.binary.resolve(strict=True)
