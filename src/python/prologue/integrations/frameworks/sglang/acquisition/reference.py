"""Expose native source facts and protocol contracts to independent verification."""


from prologue.integrations.frameworks.sglang.shared.configuration import CaptureInput, ServingInput
from prologue.integrations.frameworks.sglang.shared.kv_layout import KvBufferAddresses, MhaKvLayout
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.resources import SourceLayout
from prologue.integrations.runtimes.pytorch.shared import CpuProtocolWork, CpuTransfer

capture_input = CaptureInput
serving_input = ServingInput
cpu_protocol = CpuProtocolWork
cpu_transfer = CpuTransfer
stream_declarations = PipelineABI.stream_declarations


def observed_scopes(raw, layouts):
    return SourceLayout(raw["environment"], layouts).lower(raw["scopes"])


def kv_buffers(recording):
    native = recording["registration"]
    return KvBufferAddresses(MhaKvLayout.read(recording["layout"]), recording["page_size"],
                             native["pointers"], native["lengths"], native["page_bytes"])
