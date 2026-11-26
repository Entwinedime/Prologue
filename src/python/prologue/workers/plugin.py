"""Load the selected observation plugin only for a worker with an active session."""

import os
from pathlib import Path

from prologue.common.io import read_json
from prologue.workers.registry import entry


def install():
    """Register the installed framework observation plugin inside the native process selected by the worker."""

    session = os.environ.get("PROLOGUE_GRAPH_SESSION")
    if not session:
        return

    # The upstream plugin can load outside any capture, including a CPU-only worker.
    description = read_json(Path(session))
    entry("observation.plugin", description).install(description)
