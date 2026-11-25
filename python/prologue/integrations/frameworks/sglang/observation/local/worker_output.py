"""Bind a native worker to its device and publish its closed observation."""

from prologue.common.io import write_json
from prologue.integrations.frameworks.sglang.shared.source_layout import SourceLayout
from prologue.integrations.platforms.ascend.analysis.observation import AscendObservation
from prologue.integrations.platforms.ascend.analysis.scoped_tasks import tensor_layouts
from prologue.integrations.platforms.ascend.runtime.export import AscendExport


def activate_device(rank, world, staged):
    import torch

    if torch.npu.device_count() != (1 if staged else world):
        raise ValueError("visible Ascend device count differs from the requested placement")
    device = 0 if staged else rank
    torch.npu.set_device(device)
    return device


def publish_facts(directory, rank, raw, sdk, name):
    AscendExport.run(directory, rank, sdk)
    raw["scopes"] = SourceLayout(raw["environment"], tensor_layouts).lower(raw["scopes"])
    facts = AscendObservation(directory, rank, raw, sdk / "trace.json", evidence=False).document()
    write_json(directory / name, facts)


def publish_capture(directory, revision, starts, **evidence):
    write_json(directory / "capture.json", {
        "framework_revision": revision, "engine_starts": 0, "model_runner_starts": starts, **evidence,
    })
