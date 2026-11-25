"""Bind a native worker to its device and publish its closed observation."""

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.shared.source_layout import SourceLayout


def activate_device(rank, world, staged):

    if device_runtime().api.device_count() != (1 if staged else world):
        raise ValueError("visible device count differs from the requested placement")
    device = 0 if staged else rank
    device_runtime().api.set_device(device)
    return device


def publish_facts(directory, rank, raw, sdk, name):
    device_runtime().export.run(directory, rank, sdk)
    raw["scopes"] = SourceLayout(raw["environment"], device_runtime().tensor_layouts).lower(raw["scopes"])
    facts = device_runtime().observation(directory, rank, raw, sdk / "trace.json", evidence=False).document()
    write_json(directory / name, facts)


def publish_capture(directory, revision, starts, **evidence):
    write_json(directory / "capture.json", {
        "framework_revision": revision, "engine_starts": 0, "model_runner_starts": starts, **evidence,
    })
