"""Bind native SGLang shard geometry to the available devices."""

from pathlib import Path

from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.rules import LocalAcquisition
from prologue.host.runtime.docker import DockerImage
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.acquisition.device_samples import CostSamples
from prologue.integrations.frameworks.sglang.acquisition.host_cost_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.acquisition.local_attention import LocalAttention
from prologue.integrations.frameworks.sglang.acquisition.local_copy import LocalCopies
from prologue.integrations.frameworks.sglang.acquisition.local_embedding import LocalEmbedding
from prologue.integrations.frameworks.sglang.acquisition.local_prebuilt import LocalPrebuilt
from prologue.integrations.frameworks.sglang.acquisition.local_scheduler import (
    LocalSchedulerBatch,
    LocalWorkerBinding,
)
from prologue.integrations.frameworks.sglang.acquisition.local_work import (
    LocalDeviceWork,
    LocalHostWork,
    LocalPeerWork,
)
from prologue.integrations.frameworks.sglang.acquisition.placement import SGLangPlacement
from prologue.integrations.frameworks.sglang.acquisition.sampling import SGLangSampling
from prologue.integrations.frameworks.sglang.acquisition.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.shared.attention import AttentionCatalog


class SGLangLaunch(SGLangPlacement):
    sampling_input = staticmethod(AttentionCatalog.runner)

    @staticmethod
    def export_profile(rank, output):
        return device_runtime().export.command(rank, output)

    @staticmethod
    def interpret(rank, trace, output, *, evidence):
        return ["python3", "-m", "prologue.workers.capture", "interpret", "--input", "/source", "--rank", str(rank),
                "--trace", trace, "--output", output, *(["--evidence"] if evidence else [])]

    @staticmethod
    def sampling_cuts(rank, identities, output):
        return ["python3", "-m", "prologue.workers.capture", "sampling-cuts", "--input", "/source",
                "--rank", str(rank), "--work", "/production/work", "--identities", identities,
                "--runner", "/production/inputs/sampling.json", "--output", output]

    @staticmethod
    def host_samples(options):
        HostCostSamples(options).run()

    @staticmethod
    def device_samples(options):
        CostSamples.export(options)

    @classmethod
    def local_acquisition(cls, release, arguments):
        attention = LocalAttention(release.templates.sampling_sources if release else None)
        rules = (
            LocalWorkerBinding,
            LocalSchedulerBatch(arguments),
            LocalHostWork(arguments),
            LocalDeviceWork(arguments),
            LocalPeerWork(arguments),
            LocalCopies,
            attention,
            LocalPrebuilt,
            LocalEmbedding,
        )
        return LocalAcquisition(
            rules, cls.check_local_placement, WorkerCapture.template_resources, release.identity if release else None
        )

    def sampling(self, source):
        return SGLangSampling(self, source)

    def capture(self, arguments, limit=None, *, latency_only=False, host_timing=False, service_reference=False,
                attention_sites=None):
        if attention_sites is not None:
            ranks = {str(rank) for rank in range(self.serving.worker_count)}
            if not isinstance(attention_sites, dict) or not attention_sites or not attention_sites.keys() <= ranks:
                raise ValueError("serving comparison requires module lists keyed by participating ranks")
            for sites in attention_sites.values():
                if (
                    not isinstance(sites, list)
                    or not sites
                    or not all(isinstance(site, str) and site for site in sites)
                    or len(sites) != len(set(sites))
                ):
                    raise ValueError("each selected rank requires distinct native module names")

        model = Path(self.serving.model["path"]).resolve(strict=True)
        output = arguments.output.resolve()
        if output.is_relative_to(model):
            raise ValueError("capture output must not write inside the model directory")
        destination = str(Path(self.serving.model["path"]).absolute())
        mounts = ["--mount", f"type=bind,src={model},dst={destination},readonly"]
        image = DockerImage.inspect(arguments.source_image)
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
        return NativeAttempt(arguments, inputs, runtime, worker, mounts, self.platform_runtime(), limit)
