"""Bind a logical SGLang worker or native cut to the declared physical deployment."""

from pathlib import Path

from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain
from prologue.integrations.frameworks.sglang.shared.configuration import CaptureInput
from prologue.integrations.platforms.ascend.host.deployment import AscendDeployment
from prologue.runtime.docker import DockerImage


class SGLangAscendPlacement:
    platform_runtime = AscendDeployment
    domain = WorkerDomain
    image_revision = staticmethod(WorkerDomain.image_revision)

    def __init__(self, job, deployment):
        self.job = job
        self.serving = CaptureInput(job)
        self.platform = self.placement(deployment)
        self.local_shards = self.platform.single_device and self.serving.worker_count > 1

    def placement(self, deployment):
        return AscendDeployment(deployment, self.serving.worker_count, len(self.serving.instances) > 1)

    def require_recorded_placement(self, deployment):
        if self.placement(deployment).environment != self.platform.environment:
            raise ValueError("published capture uses a different placement from the planned source")

    def worker_options(self, *, reference):
        if self.local_shards and reference:
            raise ValueError("an independent multi-worker timing reference requires the physical target topology")
        return ["--local-shards"] if self.local_shards else []

    @classmethod
    def local_attempt(
        cls,
        arguments,
        limit,
        deployment,
        cards,
        revision,
        job,
        worker,
        *,
        runtime_trace,
        model=None,
        host_timing=False,
        **evidence,
    ):
        platform = AscendDeployment({"acquisition_mode": deployment.get("acquisition_mode", "full"),
                                     "physical_devices": cards}, len(cards), False)
        image = DockerImage.inspect(arguments.source_image)
        upstream = cls.domain.image_revision(image)
        if revision is not None and upstream != revision:
            raise EvidenceGap("framework_revision_mismatch", required=revision, image=image["Id"])
        inputs = {"job": job, "deployment": deployment, "physical_devices": platform.devices,
                  "image": image["Id"], "framework_revision": upstream, **evidence}
        mounts = []
        if model is not None:
            path = Path(model).resolve(strict=True)
            if arguments.output.resolve().is_relative_to(path):
                raise ValueError("local sample output must be outside the native model directory")
            mounts = ["--mount", f"type=bind,src={path},dst={Path(model).absolute()},readonly"]
        options = platform.options(latency_only=not runtime_trace, host_timing=host_timing)
        return NativeAttempt(arguments, inputs, options, [worker], mounts, cls.platform_runtime, limit)

    @staticmethod
    def check_local_placement(query, deployment):
        platform = AscendDeployment(deployment, 1, False)
        operand = query.get("features", {}).get("transport") if query.get("work") == "collective" else None
        if operand and len(operand["required"]) > 1 and (
            platform.single_device or max(operand["required"]) >= len(deployment["physical_devices"])
        ):
            raise EvidenceGap("missing_physical_peers", required_ranks=operand["required"],
                              physical_devices=deployment["physical_devices"])
        roles = query["domain"].get("worker_roles", {})
        ranks = sorted(map(int, roles))
        if query.get("action") == "initialize_workers":
            ranks = [
                rank for instance in query["instances"] for rank in range(instance["rank_begin"], instance["rank_end"])
            ]
        if ranks and not platform.single_device and max(ranks) >= len(deployment["physical_devices"]):
            raise EvidenceGap("missing_physical_workers", required_ranks=ranks,
                              physical_devices=deployment["physical_devices"])
