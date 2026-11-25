"""Complete the image's domain packages with Python dependencies of its public entry points."""

import ast
import shutil
import sys
from importlib.util import resolve_name
from pathlib import Path


def stage(domains, source, target, entries):
    def path_for(name):
        base = domains.joinpath(*name.split("."))
        if base.is_dir():
            relative = base.relative_to(domains) / "__init__.py"
            return relative, source / relative
        return base.with_suffix(".py").relative_to(domains), base.with_suffix(".py")

    # A clean output contains only reachable modules from the explicitly staged package directories.
    target.mkdir()
    pending, installed = list(entries), set()
    while pending:
        name = pending.pop()
        if name in installed:
            continue
        installed.add(name)
        relative, path = path_for(name)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]

        parent = name.rpartition(".")[0]
        if parent:
            pending.append(parent)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "prologue")
            elif isinstance(node, ast.ImportFrom):
                origin = resolve_name("." * node.level + (node.module or ""), package)
                if origin.split(".")[0] != "prologue":
                    continue
                pending.append(origin)
                for alias in node.names:
                    child = f"{origin}.{alias.name}"
                    if path_for(child)[1].is_file():
                        pending.append(child)

        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


if __name__ == "__main__":
    domains, source, target, *entries = sys.argv[1:]
    stage(Path(domains), Path(source), Path(target), entries)
