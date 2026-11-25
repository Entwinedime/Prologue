"""Load the selected observation plugin only for a worker with an active session."""

import os
from pathlib import Path

from prologue.common.io import read_json
from prologue.integrations.bindings import route
from prologue.workers.registry import entry


def install():
    if session := os.environ.get("PROLOGUE_GRAPH_SESSION"):
        description = read_json(Path(session))
        route().require_job(description)
        entry("observation.plugin").install(description)
