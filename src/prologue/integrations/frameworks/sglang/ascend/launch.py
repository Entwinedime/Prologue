"""Bind native SGLang shard geometry to the available Ascend devices."""

from pathlib import Path

from prologue.acquisition.capture import NativeAttempt
from prologue.integrations.frameworks.sglang.acquisition.domains import SourceDomain
from prologue.integrations.frameworks.sglang.configuration import CaptureInput
from prologue.integrations.platforms.ascend.deployment import AscendDeployment


class SGLangAscendLaunch:
    image_revision = staticmethod(SourceDomain.image_revision)

    def sampling(self, source):
        from prologue.integrations.frameworks.sglang.ascend.sampling import SGLangSampling

        return SGLangSampling(self, source)

    @staticmethod
    def source_planner(source, deployment):
        from prologue.integrations.frameworks.sglang.acquisition.planning import SourcePlanning

        return SourcePlanning(source, joint_observation=not AscendDeployment.is_single_device(deployment))

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

    def capture(self, arguments, limit=None, *, latency_only=False, host_timing=False, service_reference=False,
                attention_sites=None):
        model = Path(self.serving.model["path"]).resolve(strict=True)
        output = arguments.output.resolve()
        if output.is_relative_to(model):
            raise ValueError("capture output must not write inside the model directory")
        destination = str(Path(self.serving.model["path"]).absolute())
        mounts = ["--mount", f"type=bind,src={model},dst={destination},readonly"]
        image = NativeAttempt.image(arguments.source_image)
        inputs = {
            "job": self.job, "deployment": self.platform.source, "image": image["Id"],
            "framework_revision": self.image_revision(image), "model_path": str(model),
            "model_destination": destination, "resource_trace": arguments.resource_trace or service_reference,
        }
        if latency_only or host_timing or attention_sites:
            inputs["latency_only"] = True
        if host_timing:
            inputs["host_timing"] = True
        if service_reference:
            inputs["service_reference"] = True
        if attention_sites:
            inputs["job"] = {**self.job, "attention_sites": attention_sites}

        latency = latency_only or bool(attention_sites)
        runtime = self.platform.options(
            detailed_timing=service_reference, host_timing=host_timing, latency_only=latency,
        )
        worker = self.worker_options(reference=service_reference or latency_only)
        for flag, active in (("--resource-trace", arguments.resource_trace), ("--latency-only", latency_only),
                             ("--attention", attention_sites),
                             ("--host-timing", host_timing), ("--service-reference", service_reference)):
            if active:
                worker.append(flag)
        return NativeAttempt(arguments, inputs, runtime, worker, mounts, limit)
