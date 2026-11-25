"""Host operations and native image defaults for an SGLang ABI and platform."""

from prologue.integrations.frameworks.sglang.abi.registry import default_revision, select
from prologue.integrations.frameworks.sglang.acquisition.cost_reuse import BatchHostReuse
from prologue.integrations.frameworks.sglang.acquisition.device_samples import CostSamples
from prologue.integrations.frameworks.sglang.acquisition.host_cost_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.acquisition.launch import SGLangLaunch
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import SGLangTask
from prologue.integrations.frameworks.sglang.shared.configuration import CaptureInput


class HostAdapter:
    prediction = SGLangTask
    capture_input = CaptureInput
    native = SGLangLaunch
    samples = CostSamples
    host_samples = HostCostSamples
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
