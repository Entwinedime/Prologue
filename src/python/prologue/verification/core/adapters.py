"""Translate framework and device evidence through the adapters selected for the verification job."""

from prologue.common.io import read_json
from prologue.integrations.bindings import device_runtime, platform_backend, route


def framework_evidence():
    return route().framework_module("acquisition.evidence")


def platform_evidence():
    return platform_backend()


def stream_declarations(*args, **kwargs):
    return framework_evidence().stream_declarations(*args, **kwargs)


def observed_scopes(raw):
    return framework_evidence().observed_scopes(raw, device_runtime().tensor_layouts)


def kv_buffers(recording):
    return framework_evidence().kv_buffers(recording)


def device_coordinates(capture, graphs, rank, raw, graph):
    facts = read_json(graphs / f"{rank}-facts.json")
    sdk = platform_evidence().recording(capture, rank, raw, facts, graphs / f"{rank}-sdk/trace.json")
    platform_evidence().stream_effects(sdk).inspect(graph)
    return sdk.logical_coordinates


def device_effects(graph, capture, rank, native, facts, trace):
    evidence = platform_evidence().reference_evidence(capture, rank, native, facts, trace)
    return evidence, platform_evidence().stream_effects(evidence).inspect(graph)
