"""Bind a target serving task to matching representative native workers."""

import json
from dataclasses import asdict
from functools import cached_property

from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.shared.batches import BatchOperands
from prologue.integrations.frameworks.sglang.shared.configuration import (
    InstanceConfiguration,
    ServingInput,
)

from .contracts import WorkerDomain


class SGLangTask:
    @property
    def control_command(self):
        domain = json.dumps(asdict(platform_domain()), separators=(",", ":"))
        return ("python3", "-m", "prologue.workers.control", "--kernel-python",
                "/opt/prologue/sglang/sgl-kernel/python", "--platform-domain", domain)

    def execution_domains(self):
        return (WorkerDomain.execution_domain(w["environment"]) for i in self.instances for w in i["participants"])

    def require_control(self, image):
        revision = WorkerDomain.image_revision(image)
        if any(worker["domain"]["implementation"]["framework_revision"] != revision for worker in self.workers):
            raise ValueError("control image and primary source use different upstream SGLang revisions")

    def bind_gap(self, action, domain):
        return BatchOperands.bind(action, self.serving, domain)

    def __init__(self, source, catalog) -> None:
        if source.keys() != {"model", "instances", "workload"}:
            raise ValueError("prediction requires model, instances and workload")
        self.serving = ServingInput(source)
        structures = {i.mode: i.configuration.structure() for i in self.serving.instances}
        self.domain = {"model": self.serving.model, "instances": structures}
        self.binding = catalog.select(self.domain)

    @property
    def workers(self):
        return self.binding.headers[:self.binding.worker_count]

    def missing_workers(self):
        instances = []
        for instance in self.serving.instances:
            implementation = ["framework_revision", "model", "dtype", "attention_backend", "load_format",
                              "disaggregation_mode", "page_size", "allocator", "cache", "scheduling"]
            roles = ["parallel", "groups", "modules", "kv_layout", "expert_layers", "representative"]

            if "ep" in instance.configuration.values:
                implementation.append("expert_layout")
            if instance.mode != "null":
                roles.append("kv_peers")

            readback = {"environment": ["framework", "torch", "device"], "implementation": implementation,
                        "worker_roles": roles, "initial_state": ["logical_kv", "workspace"]}
            instances.append({"id": instance.identity, "mode": instance.mode, "rank_begin": instance.ranks.start,
                              "rank_end": instance.ranks.stop, "settings": instance.configuration.settings(),
                              "readback": readback})

        # Capacities belong to native initialization. The binding domain names reusable structure.
        return {"status": "missing_worker_binding", "action": "initialize_workers", "domain": self.domain,
                "instances": instances}

    def participants(self, description: InstanceConfiguration):
        representatives = {}
        for index, header in enumerate(self.workers):
            domain = header["domain"]
            if domain["implementation"]["disaggregation_mode"] != description.mode:
                continue

            rank = header["resources"][0]["rank"]
            if rank in representatives:
                raise ValueError(f"more than one library claims representative worker {rank}")
            representatives[rank] = index, domain

        domain = next(iter(representatives.values()))[1]
        common = WorkerDomain.role_plan(domain)
        if any(WorkerDomain.role_plan(source) != common for _, source in representatives.values()):
            raise ValueError("candidate libraries describe different native role plans or implementation environments")

        roles = domain["worker_roles"].values()
        participants = []
        for role in sorted(roles, key=lambda r: (r["parallel"]["pp_rank"], r["parallel"]["tp_rank"])):
            rank, representative = role["parallel"]["rank"], role["representative"]
            if representative not in representatives:
                raise ValueError(f"worker {rank} requires missing representative library {representative}")

            index, source = representatives[representative]
            participants.append({"id": f"{description.identity}/{rank}", "rank": rank, "template": index,
                                 "environment": WorkerDomain.bind_worker(source, role)})

        return participants

    @cached_property
    def instances(self):
        if WorkerDomain.observed(self.workers) != self.domain:
            raise ValueError("published workers do not match the target serving domain")
        instances = []
        for worker in self.serving.instances:
            config = worker.configuration
            instances.append({"id": worker.identity, "mode": worker.mode, "settings": config.settings(),
                              "control": config.control(), "expert_routes": config.routes(self.serving.workload),
                              "participants": self.participants(worker)})
        return instances

    def document(self):
        return {"instances": self.instances, "workload": self.serving.workload,
                "template_sources": self.binding.headers}
