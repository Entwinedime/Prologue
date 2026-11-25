"""Import previously compiled worker templates and fitted service costs into one portable release."""

import argparse
import json
from pathlib import Path

from prologue.acquisition.recording import PublishedCapture, SourceProductions
from prologue.libraries.release import LibraryRelease, asset_path


def import_assets(collection, costs, output):
    published = SourceProductions(collection)
    observations, identities = {}, {}
    for job, directory in published.paths.items():
        directory = directory.resolve(strict=True)
        recording = PublishedCapture(published.captures[job], published.jobs[str(job)])
        inputs = json.loads((directory / "production.json").read_bytes())["inputs"]
        if Path(inputs["capture_path"]).resolve(strict=True) != recording.source:
            raise ValueError(f"template production belongs to another native recording: {directory}")

        result = json.loads((directory / "result.json").read_bytes())
        paths = []
        for rank, name in zip(result["observed_roles"], result["templates"], strict=True):
            sampling = result.get("sampling", {}).get(str(rank))
            paths.append((asset_path(directory, name), asset_path(directory, sampling) if sampling else None))
        identity = identities[job] = recording.attempt_id
        value = recording.inputs["deployment"], paths
        if observations.setdefault(identity, value) != value:
            raise ValueError(f"conflicting template productions for recording {identity}")

    # Translate the archived collection's role declarations once; requests and capture budgets stay with the task.
    bindings = []
    jobs = [job for job in published.plan["sources"] if job["id"] in identities]
    for job in jobs:
        if not job["configurations"]:
            continue
        imports = [identities[local["id"]] for local in jobs
                   if any(row["source"] == job["id"] for row in local.get("local_consumers", []))]
        bindings.append({"domain": job["structure"], "workers": identities[job["id"]], "imports": imports})
    return LibraryRelease.publish(output, observations, bindings, costs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_assets(args.collection, args.costs, args.output)
