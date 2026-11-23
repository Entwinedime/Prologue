"""Exchange a complete command input with the installed C++ core."""

import json
import subprocess


def run_core(command, document):
    result = subprocess.run(
        ["prologue-core", command], input=json.dumps(document), encoding="utf-8", capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"{command} exited with {result.returncode}")
    return json.loads(result.stdout)
