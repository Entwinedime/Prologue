"""Select the installed framework and platform for a host task."""

from dataclasses import dataclass

from prologue.integrations.frameworks.sglang.acquisition.cost_reuse import BatchHostReuse
from prologue.integrations.frameworks.sglang.acquisition.device_samples import CostSamples
from prologue.integrations.frameworks.sglang.acquisition.host_cost_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.acquisition.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import SGLangTask
from prologue.integrations.frameworks.sglang.shared.configuration import CaptureInput


@dataclass(frozen=True, slots=True)
class Adapters:
    prediction: type
    capture_input: type
    native: type
    samples: type
    host_samples: type
    reuse: type


class AdapterRegistry:
    default = ("sglang", "ascend")
    bindings = {default: Adapters(SGLangTask, CaptureInput, SGLangAscendLaunch, CostSamples, HostCostSamples, BatchHostReuse)}

    @classmethod
    def select(cls, framework=None, platform=None):
        key = (framework or cls.default[0], platform or cls.default[1])
        try:
            return cls.bindings[key]
        except KeyError:
            raise ValueError(f"unsupported framework and platform: {key}") from None
