"""Native evidence operations and shared ABI types used by independent verification."""

from prologue.runtime.io import read_json
from prologue.integrations.frameworks.sglang.shared.configuration import CaptureInput, ServingInput
from prologue.integrations.frameworks.sglang.shared.kv_layout import KvBufferAddresses, MhaKvLayout
from prologue.integrations.frameworks.sglang.shared.pipeline import PipelineABI
from prologue.integrations.frameworks.sglang.shared.source_layout import SourceLayout
from prologue.integrations.platforms.ascend.analysis.clock import DeviceCalibration
from prologue.integrations.platforms.ascend.analysis.host_alignment import ThreadService
from prologue.integrations.platforms.ascend.analysis.scoped_tasks import tensor_layouts
from prologue.integrations.platforms.ascend.analysis.services import ObservedServices
from prologue.integrations.platforms.ascend.analysis.trace import Interval, ThreadIntervals
from prologue.integrations.platforms.ascend.reference.recording import SdkRecording
from prologue.integrations.platforms.ascend.reference.reference import NativeEffectEvidence
from prologue.integrations.platforms.ascend.reference.stream_effects import StreamEffects
from prologue.integrations.runtimes.pytorch.protocol_work import CpuProtocolWork, CpuTransfer

stream_declarations = PipelineABI.stream_declarations


def observed_scopes(raw):
    return SourceLayout(raw["environment"], tensor_layouts).lower(raw["scopes"])


def kv_buffers(recording):
    native = recording["registration"]
    return KvBufferAddresses(MhaKvLayout.read(recording["layout"]), recording["page_size"],
                             native["pointers"], native["lengths"], native["page_bytes"])


def device_coordinates(capture, graphs, rank, raw, graph):
    facts = read_json(graphs / f"{rank}-facts.json")
    sdk = SdkRecording(capture, rank, raw, facts, graphs / f"{rank}-sdk/trace.json")
    StreamEffects(sdk).inspect(graph)
    return sdk.logical_coordinates


def device_effects(graph, capture, rank, native, facts, trace):
    evidence = NativeEffectEvidence(capture, rank, native, facts, trace)
    return evidence, StreamEffects(evidence).inspect(graph)


__all__ = ['CaptureInput', 'CpuProtocolWork', 'CpuTransfer', 'DeviceCalibration', 'Interval', 'ObservedServices', 'ServingInput', 'ThreadIntervals', 'ThreadService', 'device_coordinates', 'device_effects', 'kv_buffers', 'observed_scopes', 'stream_declarations']
