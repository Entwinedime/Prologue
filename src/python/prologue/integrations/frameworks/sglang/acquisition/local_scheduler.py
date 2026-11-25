"""Qualify worker and batch queries against an initialized native scheduler."""

import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.io import write_json
from prologue.host.acquisition.rules import EvidenceGap, missing_root
from prologue.integrations.frameworks.sglang.acquisition.batch_calibration import BatchCalibration
from prologue.integrations.frameworks.sglang.acquisition.batch_readback import BatchReadback
from prologue.integrations.frameworks.sglang.acquisition.local_base import LocalRule
from prologue.integrations.frameworks.sglang.acquisition.placement import SGLangAscendPlacement
from prologue.integrations.frameworks.sglang.acquisition.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain
from prologue.integrations.frameworks.sglang.shared.attention import AttentionCatalog
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration
from prologue.integrations.frameworks.sglang.shared.ingress import BatchOperands
from prologue.integrations.platforms.ascend.analysis.scoped_tasks import ScopedTasks
from prologue.integrations.platforms.ascend.host.deployment import AscendDeployment


class LocalWorkerBinding(LocalRule):
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

    @classmethod
    def prepare(cls, queries, deployment, arguments, limit):
        first = cls.common_session(queries, deployment, "worker_initial_state")
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
        return SGLangAscendPlacement.local_attempt(arguments, limit, deployment, placement.devices, None, job,
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
        if observed != query["domain"] or any(
            set(fields) - sections[key].keys() for key, fields in instance["readback"].items()
        ):
            raise ValueError("native worker does not satisfy the requested readback")
        if (
            capacity["page_size"] != settings["page_size"]
            or asset["configured_settings"] != settings
            or initial["logical_kv"] != "empty_after_native_flush"
            or initial["free_slots"] != capacity["slot_capacity"]
            or initial["free_tokens"] != capacity["kv_pages"] * capacity["page_size"]
            or initial["workspace"]
            != {"prefill_tokens": [1], "decode_steps": 1, "device_drained": True, "device_cache_emptied": False}
        ):
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
            raise ValueError(
                "the initialized group must observe every native representative and bind every logical rank"
            )
        name = "workers/binding.json"
        product = assets[0] if len(assets) == 1 else {"participants": assets}
        return {name: product}, [
            {"status": "qualified", "asset": name, "ranks": [w.rank for w in workers]} for _ in queries
        ]


class LocalSchedulerBatch(LocalRule):
    name = BatchReadback.rule

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
        session = cls.common_session(queries, deployment, "worker", "batch_operands")
        domain, cut, separate, inputs, model, queues = session
        phase, members = cut
        identities = [row["id"] for row in inputs["requests"]]
        if len(set(identities)) != len(identities) or identities != [member["id"] for member in members]:
            raise ValueError("local inputs must name the requested members once in their original order")
        prefix = cls.priming(phase, members, separate, inputs["requests"])
        configuration = NativeConfiguration(inputs["configuration"])
        requested = WorkerDomain.observed([{"domain": domain}])
        if configuration.structure() != requested["instances"]["null"]:
            raise ValueError("supplied native configuration differs from the requested worker")
        job = {
            "configuration": configuration.values,
            "model": requested["model"],
            "batch": {
                "cut": "model" if model else "host",
                "phase": phase,
                "requests": inputs["requests"],
                "priming": prefix,
            },
        }
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
        self.binary(self.arguments, "missing_template_compiler")
        job, devices, revision = self.job(queries, deployment)
        calibration = BatchCalibration.cuts(job, queries[0]["parameters"].get("sequence_capacity"))
        if calibration:
            job["calibration"] = calibration
        return SGLangAscendPlacement.local_attempt(
            arguments,
            limit,
            deployment,
            devices,
            revision,
            job,
            "--local-worker",
            runtime_trace=True,
            host_timing=bool(calibration),
            model=job["model"]["path"],
        )

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
        model = self.compiler(self.arguments, timeout, "missing_template_compiler")
        products, answers = self.select(queries, worker, facts, bounds["phase"], asset, model)
        if products and "calibration" in recording.inputs["job"]:
            support = BatchCalibration.products(recording, worker.rank, model)
            for answer in answers:
                if answer["status"] == "qualified":
                    answer["support"] = sorted(support)
            products.update(support)
        return products, answers

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
                sampling = AttentionCatalog.from_catalog(worker.source, worker.source / "sdk/trace.json",
                                                      worker.rank, catalog, evidence, runner, ScopedTasks.source)
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
