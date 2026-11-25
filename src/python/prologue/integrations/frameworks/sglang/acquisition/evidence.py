"""Translate native SGLang evidence into the independent verifier's input contracts."""

from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.configuration import (
    CaptureInput,
    KvBufferAddresses,
    MhaKvLayout,
    ServingInput,
)
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
