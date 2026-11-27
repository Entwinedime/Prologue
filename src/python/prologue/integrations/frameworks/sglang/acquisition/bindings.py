"""Bind one SGLang ABI and placement to native host operations."""

from pathlib import Path

from prologue.common.components import contributions
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.rules import AcquisitionContext, LocalAcquisition
from prologue.host.runtime.docker import DockerImage
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.abi.registry import default_revision, select
from prologue.integrations.frameworks.sglang.shared.configuration import (
    AttentionCatalog,
    CaptureInput,
)
from prologue.integrations.frameworks.sglang.shared.resources import SourceLayout

from . import recipes
from .contracts import SGLangPlacement
from .evidence import CostSamples
from .host_samples import HostCostSamples
from .sampling import BatchHostReuse, SGLangSampling
from .target import SGLangTask


class SGLangLaunch(SGLangPlacement):
    """Resolve the logical workload and rank placement before assigning any concrete device invocation."""

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

    @classmethod
    def local_acquisition(cls, release, arguments, *, training_sources=None):
        context = AcquisitionContext(
            arguments, release.identity if release else None,
            release.templates.sampling_sources if release else training_sources,
        )
        installed = contributions(recipes, "rules")
        rules = (kind(context) for kind in installed if training_sources is None or kind.source_window)
        return LocalAcquisition(rules, context, cls.check_local_placement, SourceLayout.resource_bindings)

    def sampling(self, source):
        return SGLangSampling(self, source)

    def capture(self, arguments, limit=None, *, latency_only=False, batch_timing=False, host_timing=False,
                service_reference=False, attention_sites=None):
        if attention_sites is not None:
            ranks = {str(rank) for rank in range(self.serving.worker_count)}
            if not isinstance(attention_sites, dict) or not attention_sites or not attention_sites.keys() <= ranks:
                raise ValueError("serving comparison requires module lists keyed by participating ranks")

            for sites in attention_sites.values():
                if (not isinstance(sites, list) or not sites or not all(isinstance(s, str) and s for s in sites)
                        or len(sites) != len(set(sites))):
                    raise ValueError("each selected rank requires distinct native module names")

        destination = Path(self.serving.model["path"]).absolute()
        mounts = self.model_mount(destination, arguments.output)
        image = DockerImage.inspect(arguments.source_image)
        inputs = {
            "job": self.job, "deployment": self.platform.source, "image": image["Id"],
            "framework_revision": self.image_revision(image), "model_path": str(destination.resolve(strict=True)),
            "model_destination": str(destination), "resource_trace": arguments.resource_trace or service_reference,
        }
        latency = latency_only or batch_timing or bool(attention_sites)
        flags = {"resource_trace": arguments.resource_trace, "latency_only": latency_only,
                 "batch_timing": batch_timing, "attention": attention_sites, "host_timing": host_timing,
                 "service_reference": service_reference}

        if attention_sites:
            inputs["job"] = self.job | {"attention_sites": attention_sites}
        inputs.update({k: True for k in ("batch_timing", "host_timing", "service_reference") if flags[k]})
        if latency or host_timing:
            inputs["latency_only"] = True

        runtime = self.platform.options(latency_only=latency, detailed_timing=service_reference,
                                        host_timing=host_timing)
        worker = self.worker_options(reference=service_reference or latency)
        worker.extend("--" + key.replace("_", "-") for key, active in flags.items() if active)
        return NativeAttempt(arguments, inputs, runtime, worker, mounts, self.platform_runtime(), limit)


class HostAdapter:
    """Supply the framework recipes and production contracts selected by the explicit host route."""

    prediction = SGLangTask
    capture_input = CaptureInput
    native = SGLangLaunch
    samples = CostSamples
    host_samples = staticmethod(HostCostSamples.export)
    device_samples = staticmethod(CostSamples.export)
    reuse = BatchHostReuse

    @staticmethod
    def revision(value):
        return select(value).revision

    @staticmethod
    def images(selection):
        suffix = "" if selection.revision == default_revision else "-" + selection.revision[:8]
        tag = selection.platform_module().platform.image_tag
        return {"source_image": f"prologue-{selection.framework}{suffix}:{tag}",
                "control_image": f"prologue-control{suffix}:{tag}"}
