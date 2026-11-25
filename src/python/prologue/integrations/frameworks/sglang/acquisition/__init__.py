"""Host contracts for native worker domains, placement, recording and local acquisition."""

from dataclasses import dataclass
from pathlib import Path

from prologue.common.io import read_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.rules import EvidenceGap, missing_root
from prologue.host.libraries.release import asset_path
from prologue.host.runtime.docker import DockerImage, ModelingProcess
from prologue.integrations.bindings import RuleName, device_runtime, platform_backend, platform_domain, route
from prologue.integrations.frameworks.sglang.shared.configuration import (
    CaptureInput,
    NativeConfiguration,
    WorkerRoles,
)


class WorkerDomain(WorkerRoles):
    """Identify the serving structure and verify the worker environment used by native control."""

    @staticmethod
    def execution_domain(environment, state=("initial_state", "closure")):
        domain = {k: v for k, v in environment.items() if k != "device_index" and k not in state}
        domain["parallel"] = {k: v for k, v in environment["parallel"].items() if k != "rank"}
        return domain

    @classmethod
    def local_environment(cls, environment):
        return cls.execution_domain(environment, ()), environment["parallel"]["rank"], environment["device_index"]

    @classmethod
    def role_plan(cls, domain):
        return {key: value for key, value in domain.items() if key not in cls.role_fields}

    @staticmethod
    def native_domain(domain):
        return domain.get("framework") == "sglang" and platform_domain().runtime_domain_key in domain

    @classmethod
    def dense_worker(cls, domain):
        return (cls.native_domain(domain) and not domain["expert_layers"]
                and domain["implementation"]["disaggregation_mode"] == "null"
                and domain["parallel"]["logical_pp"] == domain["parallel"]["logical_ep"] == 1)

    @staticmethod
    def image_revision(image: dict) -> str:
        labels = image["Config"]["Labels"] or {}
        revision = NativeConfiguration.upstream_revision(labels.get("org.prologue.sglang.revision"))
        selection = route()
        if revision != selection.revision:
            raise ValueError("selected framework ABI and native image revision differ")
        image_env = dict(entry.split("=", 1) for entry in image["Config"].get("Env", ()))
        for field, expected in (("PROLOGUE_FRAMEWORK", selection.framework), ("PROLOGUE_PLATFORM", selection.platform)):
            if field in image_env and image_env[field] != expected:
                raise ValueError(f"native image {field} differs from the selected adapter")
        return revision

    @staticmethod
    def observed(libraries: list[dict]) -> dict:
        result = {"instances": {}}
        for library in libraries:
            domain = library["domain"]
            if domain["framework"] != "sglang":
                raise ValueError("source production has no SGLang implementation domain")
            implementation, parallel = domain["implementation"], domain["parallel"]
            model = {"path": implementation["model"]["_name_or_path"],
                     "dtype": implementation["dtype"].removeprefix("torch.")}
            if result.setdefault("model", model) != model:
                raise ValueError("source production contains different models")
            structure = {name: parallel["logical_" + name] for name in ("tp", "pp", "ep")}
            structure |= {k: implementation[k] for k in ("attention_backend", "page_size")}
            structure |= {"prefix_cache": not implementation["cache"]["disabled"], **implementation["scheduling"]}
            mode = implementation["disaggregation_mode"]
            if result["instances"].setdefault(mode, structure) != structure:
                raise ValueError("source representatives disagree about their instance structure")
        return result


class SGLangPlacement:
    image_revision = staticmethod(WorkerDomain.image_revision)

    @staticmethod
    def platform_runtime():
        return platform_backend().deployment

    def __init__(self, job, deployment):
        self.job = job
        self.serving = CaptureInput(job)
        self.platform = self.placement(deployment)
        self.local_shards = self.platform.single_device and self.serving.worker_count > 1

    def placement(self, deployment):
        return platform_backend().deployment(deployment, self.serving.worker_count, len(self.serving.instances) > 1)

    def require_recorded_placement(self, deployment):
        if self.placement(deployment).environment != self.platform.environment:
            raise ValueError("published capture uses a different placement from the planned source")

    def worker_options(self, *, reference):
        if self.local_shards and reference:
            raise ValueError("an independent multi-worker timing reference requires the physical target topology")
        return ["--local-shards"] if self.local_shards else []

    @staticmethod
    def model_mount(model, output):
        path = Path(model).resolve(strict=True)
        if output.resolve().is_relative_to(path):
            raise ValueError("capture output must be outside the native model directory")
        return ["--mount", f"type=bind,src={path},dst={Path(model).absolute()},readonly"]

    @staticmethod
    def check_local_placement(query, deployment):
        platform = platform_backend().deployment(deployment, 1, False)
        operand = query.get("features", {}).get("transport") if query.get("work") == "collective" else None
        if operand and len(operand["required"]) > 1 and (
            platform.single_device or max(operand["required"]) >= len(deployment["physical_devices"])
        ):
            raise EvidenceGap("missing_physical_peers", required_ranks=operand["required"],
                              physical_devices=deployment["physical_devices"])
        ranks = sorted(map(int, query["domain"].get("worker_roles", {})))
        if query.get("action") == "initialize_workers":
            ranks = [r for i in query["instances"] for r in range(i["rank_begin"], i["rank_end"])]
        if ranks and not platform.single_device and max(ranks) >= len(deployment["physical_devices"]):
            raise EvidenceGap("missing_physical_workers", required_ranks=ranks,
                              physical_devices=deployment["physical_devices"])


@dataclass
class LocalCut:
    """Inputs for one native probe. The rule chooses its operands and the device owns its timing."""

    job: dict
    cards: list
    revision: str | None
    model: str | None = None
    evidence: dict | None = None
    worker: str | None = None
    host_timing: bool = False

    @classmethod
    def workers(cls, job, deployment, tp, *, domain=None):
        placement = platform_backend().deployment(deployment, tp, False)
        if tp > 1:
            job["single_device"] = placement.single_device
            if domain is not None:
                job["batch"]["rank"] = WorkerDomain.rank(domain)
        revision = domain["implementation"]["framework_revision"] if domain is not None else None
        return cls(job, placement.devices, revision, job["model"]["path"])


class LocalRule:
    worker = "--local-worker"
    runtime_trace = True
    conditions = None

    def __init__(self, arguments=None):
        self.arguments = arguments

    @staticmethod
    def session_key(query, deployment):
        return query

    @classmethod
    def common_session(cls, queries, deployment):
        identity = cls.session_key(queries[0], deployment)
        if cls.conditions and any(cls.session_key(q, deployment) != identity for q in queries[1:]):
            raise EvidenceGap("different_sessions_required", conditions=list(cls.conditions))
        return identity

    @staticmethod
    def missing_action(query, action):
        return query.get("action") == action and missing_root(query) and WorkerDomain.native_domain(query["domain"])

    def prepare(self, queries, deployment, arguments, limit):
        session = self.common_session(queries, deployment)
        cut = self.cut(queries, session, deployment, arguments)
        selection = {"acquisition_mode": deployment.get("acquisition_mode", "full"), "physical_devices": cut.cards}
        platform = platform_backend().deployment(selection, len(cut.cards), False)
        image = DockerImage.inspect(arguments.source_image)
        revision = WorkerDomain.image_revision(image)
        if cut.revision is not None and revision != cut.revision:
            raise EvidenceGap("framework_revision_mismatch", required=cut.revision, image=image["Id"])
        inputs = {"job": cut.job, "deployment": deployment, "physical_devices": platform.devices,
                  "image": image["Id"], "framework_revision": revision, **(cut.evidence or {})}
        mounts = SGLangPlacement.model_mount(cut.model, arguments.output) if cut.model is not None else []
        options = platform.options(latency_only=not self.runtime_trace, host_timing=cut.host_timing)
        return NativeAttempt(arguments, inputs, options, [cut.worker or self.worker], mounts,
                             platform_backend().deployment, limit)

    @staticmethod
    def asset(name, product, answers):
        return ({name: product} if any(a["status"] == "qualified" for a in answers) else {}), answers

    @staticmethod
    def recording_part(recording, key):
        return read_json(asset_path(recording.source, recording.manifest[key]))

    @staticmethod
    def physical_card(domain, deployment):
        workers = max(map(int, domain["worker_roles"])) + 1
        placement = platform_backend().deployment(deployment, workers, False)
        return placement.card(WorkerDomain.rank(domain))

    def binary(self):
        if self.arguments.binary is None:
            raise EvidenceGap(self.compiler_gap, required="--binary")
        return self.arguments.binary.resolve(strict=True)

    def compiler(self, seconds):
        options = self.arguments
        return ModelingProcess(self.binary(), options.modeling_image,
                               options.timeout if seconds is None else min(options.timeout, seconds))


scheduler_batch_rule = RuleName("scheduler_batch")


class CopyOperand:
    @staticmethod
    def runtime(domain):
        return {k: domain[k] for k in ("device", "torch", platform_domain().runtime_domain_key)}

    @staticmethod
    def operand(definition, interval, transport):
        work = definition["work"][interval]
        copy = work.get("copy")
        if work["kind"] != "transfer" or not copy or copy["caller"] not in device_runtime().copy_sampler.directions:
            return None
        service = device_runtime().copy_sampler.service(copy["caller"])
        resource = definition["resources"][work["resource"]]
        if (transport != service or any(copy[key] != service[key] for key in ("implementation", "direction"))
                or [resource["kind"], resource["role"]] != service["resource"]):
            return None
        return copy["caller"], copy["tensor"]["dtype"], copy["element_bytes"]

    @classmethod
    def matches(cls, query):
        return (query.get("work") == "transfer" and WorkerDomain.native_domain(query["domain"])
                and cls.operand(query["definition"], query["interval"], query["features"].get("transport")) is not None)

    @classmethod
    def point(cls, query):
        features = query["features"]
        caller, dtype, width = cls.operand(query["definition"], query["interval"], features["transport"])
        size = features["axes"].get("transfer.bytes")
        if (features["phase"] or set(features["axes"]) != {"transfer.bytes"} or type(width) is not int or width <= 0
                or type(size) is not int or size <= 0 or size % width):
            raise EvidenceGap("unbound_copy_operand", caller=caller, dtype=dtype, axes=features["axes"])
        return caller, dtype, width, size


embedding_rule = "sglang.vocab_parallel_embedding"
