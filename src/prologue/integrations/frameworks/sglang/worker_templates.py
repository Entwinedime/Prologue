"""Assemble native participants from source-authorized roles, without importing SGLang."""

from __future__ import annotations

from prologue.integrations.frameworks.sglang.configuration import InstanceConfiguration, NativeConfiguration, ServingInput


class WorkerDomain:
    """Identify the serving structure and verify the worker environment used by native control."""

    role_fields = ("parallel", "kv_layout", "expert_layers", "kv_peers")

    @classmethod
    def bind_worker(cls, domain, role):
        return domain | {name: role[name] for name in cls.role_fields if name in role}

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
        return {"status": "missing_worker_binding", "action": "initialize_workers", "domain": self.domain}

    def participants(self, description: InstanceConfiguration) -> list[dict]:
        candidates = [
            (index, header) for index, header in enumerate(self.workers)
            if header["domain"]["implementation"]["disaggregation_mode"] == description.mode
        ]
        domain = candidates[0][1]["domain"]
        common = {name: value for name, value in domain.items() if name not in WorkerDomain.role_fields}
        representatives = {}
        for index, library in candidates:
            if {name: value for name, value in library["domain"].items() if name not in WorkerDomain.role_fields} != common:
                raise ValueError("candidate libraries describe different native role plans or implementation environments")
            rank = library["resources"][0]["rank"]
            if rank in representatives:
                raise ValueError(f"more than one library claims representative worker {rank}")
            representatives[rank] = index

        roles = sorted(domain["worker_roles"].values(), key=lambda role: (role["parallel"]["pp_rank"], role["parallel"]["tp_rank"]))
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
