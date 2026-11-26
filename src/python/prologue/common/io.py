"""Shared atomic JSON publication for task and observation artifacts."""

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def read_jsonl(path: Path) -> Iterator[Any]:
    with path.open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def discard_path(path: Path) -> None:
    if not path.is_symlink() and path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def content_id(value: Any) -> str:
    """Identify JSON values independently of object-field order and formatting."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def write_json(path: Path, value: Any, *, indent: int | None = None, replace: bool = True) -> None:
    """Publish a complete document or leave the destination at its previous state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=indent, separators=(",", ":") if indent is None else None, allow_nan=False)
            stream.write("\n")

        # Immutable artifacts use an atomic link; an existing file or symlink must remain untouched.
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
