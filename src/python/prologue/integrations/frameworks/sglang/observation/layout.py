"""Describe model geometry, worker roles and the observed state of native resources."""

from __future__ import annotations

from prologue.common.io import read_json, write_json
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.observation.kv_evidence import NativeKvFacts
from prologue.integrations.frameworks.sglang.shared.configuration import (
    AttentionCatalog,
)
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.resources import SourceLayout


def observed_graph(directory, rank, recording, trace, *, evidence):
    runtime = device_runtime()
    recording["scopes"] = SourceLayout(recording["environment"], runtime.tensor_layouts).lower(recording["scopes"])
    return runtime.observation(directory, rank, recording, trace, evidence=evidence)


def serving_facts(directory, rank, trace, *, evidence):
    native = NativeKvFacts(directory, rank)
    recording = native.document()
    observation = observed_graph(directory, rank, recording, trace, evidence=evidence)
    observation.name_streams(*PipelineABI.stream_declarations(recording))
    facts = observation.document()
    native.connect_metadata(facts)
    if evidence and (timing := native.protocol_timings()) is not None:
        facts["measurements"]["pd_protocol"] = timing
    return facts


def publish_cuts(source, work, rank, identities, runner, output):
    catalog = AttentionCatalog.extract(
        source, work, rank, identities, read_json(runner), device_runtime().scoped_tasks.source
    )
    write_json(output, catalog)
