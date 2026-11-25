"""Export a trained attention cut from a closed native profiling window."""

from prologue.common.io import read_json, write_json
from prologue.integrations.frameworks.sglang.shared.attention import AttentionCatalog
from prologue.integrations.platforms.ascend.analysis.scoped_tasks import ScopedTasks


def publish_cuts(source, work, rank, identities, runner, output):
    catalog = AttentionCatalog.extract(source, work, rank, identities, read_json(runner), ScopedTasks.source)
    write_json(output, catalog)
