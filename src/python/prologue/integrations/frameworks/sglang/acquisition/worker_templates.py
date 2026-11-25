"""Assemble native participants from source-authorized roles, without importing SGLang."""

from __future__ import annotations

import json
from dataclasses import asdict

from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.shared.configuration import (
    InstanceConfiguration,
    NativeConfiguration,
    ServingInput,
)
from prologue.integrations.frameworks.sglang.shared.groups import WorkerRoles
from prologue.integrations.frameworks.sglang.shared.ingress import BatchOperands


class WorkerDomain(WorkerRoles):
    """Identify the serving structure and verify the worker environment used by native control."""

    @staticmethod
    def execution_domain(environment):
        domain = {k: v for k, v in environment.items() if k not in {"device_index", "initial_state", "closure"}}
        domain["parallel"] = {k: v for k, v in domain["parallel"].items() if k != "rank"}
        return domain

    @staticmethod
    def image_revision(image: dict) -> str:
        labels = image["Config"]["Labels"] or {}
        return NativeConfiguration.upstream_revision(labels.get("org.prologue.sglang.revision"))

    @staticmethod
    def require_control(image: dict, libraries: list[dict]) -> None:
        revision = WorkerDomain.image_revision(image)
        if any(library["domain"]["implementation"]["framework_revision"] != revision for library in libraries):
            raise ValueError("control image and primary source use different upstream SGLang revisions")

    @staticmethod
    def requested(serving: ServingInput) -> dict:
        return {
            "model": serving.model,
            "instances": {
                instance.mode: instance.configuration.structure() for instance in serving.instances
            },
        }

    @staticmethod
    def observed(libraries: list[dict]) -> dict:
        result = {"instances": {}}
        for library in libraries:
            domain = library["domain"]
            if domain["framework"] != "sglang":
                raise ValueError("source production has no SGLang implementation domain")
            implementation, parallel = domain["implementation"], domain["parallel"]
            model = {
                "path": implementation["model"]["_name_or_path"],
                "dtype": implementation["dtype"].removeprefix("torch."),
            }
            if result.setdefault("model", model) != model:
                raise ValueError("source production contains different models")
            structure = {
                "tp": parallel["logical_tp"], "pp": parallel["logical_pp"], "ep": parallel["logical_ep"],
                "attention_backend": implementation["attention_backend"], "page_size": implementation["page_size"],
                "prefix_cache": not implementation["cache"]["disabled"], **implementation["scheduling"],
            }
            mode = implementation["disaggregation_mode"]
            if result["instances"].setdefault(mode, structure) != structure:
                raise ValueError("source representatives disagree about their instance structure")
        return result


class SGLangTask:
    @property
    def control_command(self):
        domain = json.dumps(asdict(platform_domain()), separators=(",", ":"))
        return ("python3", "-m", "prologue.workers.control", "--kernel-python",
                "/opt/prologue/sglang/sgl-kernel/python", "--platform-domain", domain)

    def execution_domains(self):
        for instance in self.serving.instances:
            for worker in self.participants(instance):
                yield WorkerDomain.execution_domain(worker["environment"])

    def require_control(self, image):
        WorkerDomain.require_control(image, self.workers)

    def bind_gap(self, action, domain):
        return BatchOperands.bind(action, self.serving, domain)

    def __init__(self, source, catalog) -> None:
        if source.keys() != {"model", "instances", "workload"}:
            raise ValueError("prediction requires model, instances and workload")
        self.serving = ServingInput(source)
        self.domain = WorkerDomain.requested(self.serving)
        self.binding = catalog.select(self.domain)

    @property
    def workers(self):
        return self.binding.headers[:self.binding.worker_count]

    def missing_workers(self):
        instances = []
        for instance in self.serving.instances:
            implementation = ["framework_revision", "model", "dtype", "attention_backend", "load_format",
                              "disaggregation_mode", "page_size", "allocator", "cache", "scheduling"]
            if "ep" in instance.configuration.values:
                implementation.append("expert_layout")
            role_fields = ["parallel", "groups", "modules", "kv_layout", "expert_layers", "representative"]
            if instance.mode != "null":
                role_fields.append("kv_peers")

            instances.append({
                "id": instance.identity, "mode": instance.mode,
                "rank_begin": instance.ranks.start, "rank_end": instance.ranks.stop,
                "settings": instance.configuration.settings(),
                "readback": {
                    "environment": ["framework", "torch", "device"], "implementation": implementation,
                    "worker_roles": role_fields, "initial_state": ["logical_kv", "workspace"],
                },
            })

        # The domain identifies reusable structure. Target capacities are inputs to preparing a native instance.
        return {
            "status": "missing_worker_binding", "action": "initialize_workers", "domain": self.domain,
            "instances": instances,
        }

    def participants(self, description: InstanceConfiguration) -> list[dict]:
        candidates = [
            (index, header) for index, header in enumerate(self.workers)
            if header["domain"]["implementation"]["disaggregation_mode"] == description.mode
        ]
        domain = candidates[0][1]["domain"]
        common = {name: value for name, value in domain.items() if name not in WorkerDomain.role_fields}
        representatives = {}
        for index, library in candidates:
            if {
                name: value for name, value in library["domain"].items() if name not in WorkerDomain.role_fields
            } != common:
                raise ValueError(
                    "candidate libraries describe different native role plans or implementation environments"
                )
            rank = library["resources"][0]["rank"]
            if rank in representatives:
                raise ValueError(f"more than one library claims representative worker {rank}")
            representatives[rank] = index

        roles = sorted(
            domain["worker_roles"].values(), key=lambda role: (role["parallel"]["pp_rank"], role["parallel"]["tp_rank"])
        )
        participants = []
        for role in roles:
            rank = role["parallel"]["rank"]
            representative = role["representative"]
            if representative not in representatives:
                raise ValueError(f"worker {rank} requires missing representative library {representative}")
            index = representatives[representative]

            # Assembly only reads these facts; each worker replaces top-level bindings without cloning shared trees.
            environment = WorkerDomain.bind_worker(self.binding.headers[index]["domain"], role)
            participants.append({
                "id": f"{description.identity}/{rank}", "rank": rank, "template": index, "environment": environment,
            })
        return participants

    def document(self) -> dict:
        if WorkerDomain.observed(self.workers) != self.domain:
            raise ValueError("published workers do not match the target serving domain")

        instances = []
        for description in self.serving.instances:
            configuration = description.configuration
            instances.append({
                "id": description.identity, "mode": description.mode, "settings": configuration.settings(),
                "control": configuration.control(), "expert_routes": configuration.routes(self.serving.workload),
                "participants": self.participants(description),
            })
        return {
            "instances": instances, "workload": self.serving.workload,
            "template_sources": self.binding.headers,
        }
