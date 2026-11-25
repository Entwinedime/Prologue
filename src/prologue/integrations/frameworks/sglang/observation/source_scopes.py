"""Project the serving recording with the model layout, stream declarations and native peer evidence."""

from prologue.integrations.frameworks.sglang.observation.kv_facts import NativeKvFacts
from prologue.integrations.frameworks.sglang.shared.pipeline import PipelineABI
from prologue.integrations.frameworks.sglang.shared.source_layout import SourceLayout
from prologue.integrations.platforms.ascend.observation import AscendObservation
from prologue.integrations.platforms.ascend.scoped_tasks import tensor_layouts


def serving_facts(directory, rank, trace, *, evidence):
    native = NativeKvFacts(directory, rank)
    recording = native.document()
    recording["scopes"] = SourceLayout(recording["environment"], tensor_layouts).lower(recording["scopes"])

    # Framework closure rules stay separate from runtime interpretation and the common graph projection.
    observation = AscendObservation(
        directory, rank, recording, trace, evidence=evidence
    )
    observation.name_streams(*PipelineABI.stream_declarations(recording))
    facts = observation.document()
    native.connect_metadata(facts)
    if evidence and (timing := native.protocol_timings()) is not None:
        facts["measurements"]["pd_protocol"] = timing
    return facts
