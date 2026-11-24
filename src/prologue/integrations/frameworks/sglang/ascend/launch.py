"""Bind native SGLang shard geometry to the available Ascend devices."""

from pathlib import Path

from prologue.acquisition.capture import NativeAttempt
from prologue.integrations.frameworks.sglang.ascend.attention_cuts import AttentionCuts
from prologue.integrations.frameworks.sglang.configuration import CaptureInput
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.deployment import AscendDeployment
from prologue.integrations.platforms.ascend.export import AscendExport


class SGLangAscendLaunch:
    domain = WorkerDomain
    export_profile = staticmethod(AscendExport.command)
    sampling_input = staticmethod(AttentionCuts.runner)

    @classmethod
    def local_acquisition(cls, release=None):
        from prologue.acquisition.rules import LocalAcquisition
        from prologue.integrations.frameworks.sglang.ascend.local_attention import LocalAttention
        from prologue.integrations.frameworks.sglang.ascend.local_copy import LocalCopies
        from prologue.integrations.frameworks.sglang.ascend.local_embedding import LocalEmbedding
        from prologue.integrations.frameworks.sglang.ascend.local_prebuilt import LocalPrebuilt

        attention = LocalAttention(release.templates.sampling_sources if release else None)
        return LocalAcquisition((LocalCopies, attention, LocalPrebuilt, LocalEmbedding), cls.check_local_placement,
                                release.identity if release else None)

    @classmethod
    def local_attempt(cls, arguments, limit, deployment, cards, revision, job, worker, *, runtime_trace, model=None, **evidence):
        from prologue.acquisition.rules import EvidenceGap

        platform = AscendDeployment({"acquisition_mode": deployment.get("acquisition_mode", "full"),
                                     "physical_devices": cards}, len(cards), False)
        image = NativeAttempt.image(arguments.source_image)
        if cls.domain.image_revision(image) != revision:
            raise EvidenceGap("framework_revision_mismatch", required=revision, image=image["Id"])
        inputs = {"job": job, "deployment": deployment, "physical_devices": platform.devices,
                  "image": image["Id"], "framework_revision": revision, **evidence}
        mounts = []
        if model is not None:
            path = Path(model).resolve(strict=True)
            if arguments.output.resolve().is_relative_to(path):
                raise ValueError("local sample output must be outside the native model directory")
            mounts = ["--mount", f"type=bind,src={path},dst={Path(model).absolute()},readonly"]
        return NativeAttempt(arguments, inputs, platform.options(latency_only=not runtime_trace), [worker], mounts, limit)

    @staticmethod
    def check_local_placement(query, deployment):
        from prologue.acquisition.rules import EvidenceGap

        platform = AscendDeployment(deployment, 1, False)
        if query.get("work") != "collective" or "torch_npu" not in query["domain"]:
            return
        operand = query["features"].get("transport")
        if operand and len(operand["required"]) > 1 and (
            platform.single_device or max(operand["required"]) >= len(deployment["physical_devices"])
        ):
            raise EvidenceGap("missing_physical_peers", required_ranks=operand["required"],
                              physical_devices=deployment["physical_devices"])

    def sampling(self, source):
        from prologue.integrations.frameworks.sglang.ascend.sampling import SGLangSampling

        return SGLangSampling(self, source)

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
        if attention_sites is not None:
            ranks = {str(rank) for rank in range(self.serving.worker_count)}
            if not isinstance(attention_sites, dict) or not attention_sites or not attention_sites.keys() <= ranks:
                raise ValueError("serving comparison requires module lists keyed by participating ranks")
            for sites in attention_sites.values():
                if (not isinstance(sites, list) or not sites or not all(isinstance(site, str) and site for site in sites)
                        or len(sites) != len(set(sites))):
                    raise ValueError("each selected rank requires distinct native module names")

        model = Path(self.serving.model["path"]).resolve(strict=True)
        output = arguments.output.resolve()
        if output.is_relative_to(model):
            raise ValueError("capture output must not write inside the model directory")
        destination = str(Path(self.serving.model["path"]).absolute())
        mounts = ["--mount", f"type=bind,src={model},dst={destination},readonly"]
        image = NativeAttempt.image(arguments.source_image)
        inputs = {
            "job": self.job, "deployment": self.platform.source, "image": image["Id"],
            "framework_revision": self.domain.image_revision(image), "model_path": str(model),
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
        worker = self.worker_options(reference=service_reference or latency_only or bool(attention_sites))
        for flag, active in (("--resource-trace", arguments.resource_trace), ("--latency-only", latency_only),
                             ("--attention", attention_sites),
                             ("--host-timing", host_timing), ("--service-reference", service_reference)):
            if active:
                worker.append(flag)
        return NativeAttempt(arguments, inputs, runtime, worker, mounts, limit)
