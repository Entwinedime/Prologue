"""Export a trained attention cut from a closed native profiling window."""

from prologue.common.io import read_json, write_json
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.shared.configuration import AttentionCatalog


def publish_cuts(source, work, rank, identities, runner, output):
    catalog = AttentionCatalog.extract(
        source, work, rank, identities, read_json(runner), device_runtime().scoped_tasks.source
    )
    write_json(output, catalog)
