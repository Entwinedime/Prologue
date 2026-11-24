"""Explore a task on the Docker host using pinned Prologue images."""

import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

from prologue.app.deployment import Deployment, inspect_image
from prologue.app.supplement import Supplement
from prologue.workflow.budget import Acquisition, ProcessingBudget
from prologue.workflow.exploration import Exploration


def arguments(parser):
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("data/libraries"))
    parser.add_argument("--release", default="current", help="initial release UUID or current")
    parser.add_argument("--app-image", default="prologue-app:ubuntu24.04")
    parser.add_argument("--control-image", default="prologue-scheduler:ubuntu22.04")
    parser.add_argument("--observer-image", default="prologue-observer:ubuntu22.04")
    parser.add_argument("--acquire", action="store_true", help="allow scoped additions to the shared release")
    parser.add_argument("--source", type=Path, action="append", default=[], help="independent detailed template source")
    parser.add_argument("--training", type=Path, help="independent task with one workload for structure acquisition")
    parser.add_argument("--device", type=int, help="allocated physical Ascend device for native acquisition")
    parser.add_argument("--model-directory", type=Path, help="host directory to mount at the model path")
    parser.add_argument("--attempts", type=int, default=0)
    parser.add_argument("--card-seconds", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--processing-seconds", type=int, default=900)
    parser.set_defaults(handler=explore)


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def source_identity(path):
    manifest = Acquisition.read(path / "manifest.json")
    if manifest["mode"] != "detailed":
        raise ValueError("a retained template source requires a detailed native recording")
    domain = {field: manifest[field] for field in ("framework_revision", "model", "configuration")}
    return {"path": str(path.resolve()), "manifest_sha256": digest(manifest), "domain": domain,
            "requests_sha256": digest(manifest["workload"]["requests"])}


def request(args):
    limits = {field: getattr(args, field) for field in ("attempts", "card_seconds", "max_bytes", "processing_seconds")}
    if min(limits.values()) < 0 or not args.processing_seconds or (args.device is not None and args.device < 0):
        raise ValueError("budget limits and the device index must be nonnegative, with positive processing time")
    if not args.acquire and (args.source or args.training or args.device is not None):
        raise ValueError("template sources, training and devices require --acquire")
    docs = {"task": Acquisition.read(args.task), "hardware": Acquisition.read(args.hardware)}
    if args.training:
        docs["training-task"] = Acquisition.read(args.training)

    model = docs["task"]["model"]["path"]
    directory = args.model_directory or Path(model)
    if args.device is not None and (not Path(model).is_absolute() or not directory.is_dir()):
        raise ValueError("physical acquisition requires an absolute model path and an existing host model directory")
    images = {"application": args.app_image, "control": args.control_image}
    if args.device is not None:
        images["observer"] = args.observer_image
    spec = {"store": str(args.store.resolve()), "requested_release": args.release, "image_names": images,
            "acquire": args.acquire, "sources": [source_identity(path) for path in args.source],
            "device": args.device, "model": model, "model_directory": str(directory.resolve()), "limits": limits,
            "inputs": {key: digest(document) for key, document in docs.items()}}
    return spec, docs


def pinned(output, specification, documents):
    path = output / "run.json"
    if path.exists():
        existing = Acquisition.read(path)
        if existing["request"] != specification:
            raise ValueError("the output directory is pinned to another exploration")
        return existing

    images = {role: inspect_image(name) for role, name in specification["image_names"].items()}
    control = images["control"]["framework_revision"]
    if not control or ("observer" in images and images["observer"]["framework_revision"] != control):
        raise ValueError("control and observer images must identify the same native revision")
    for key, document in documents.items():
        Acquisition.write(output / f"{key}.json", document)
    run = {"run_id": str(uuid4()), "request": specification, "images": images}
    Acquisition.write(path, run)
    return run


def prepared(deployment, command, name):
    path = deployment.output / f"{name}.json"
    if path.exists():
        return Acquisition.read(path)
    document = deployment.app(command)
    Acquisition.write(path, document)
    return document


def explore(args):
    specification, documents = request(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".explore.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = pinned(output, specification, documents)
        if (output / "result.json").exists():
            return Acquisition.read(output / "result.json")

        with ProcessingBudget(output / "processing.json", args.processing_seconds) as clock:
            deployment = Deployment(output, args.store.resolve(), {**specification, **run}, clock)
            release = prepared(deployment, ["show-release", "--store", "/library", "--release", args.release],
                               "base-release")
            plan = prepared(deployment, ["plan", "/run/task.json"], "plan")
            validate_sources(plan, specification["sources"])
            training = None
            if args.training:
                training = prepared(deployment, ["plan", "/run/training-task.json"], "training-plan")
                validate_training(plan, training)
            supplement = Supplement(deployment, training) if args.acquire else None
            return Exploration(output, plan, release["release"], deployment, supplement).run()


def validate_sources(plan, sources):
    workloads = {digest(row["requests"]) for row in plan["workloads"]}
    for source in sources:
        domain = source["domain"]
        if (domain["model"] == plan["model"] and domain["configuration"] in plan["configurations"]
                and source["requests_sha256"] in workloads):
            raise ValueError("a retained template source duplicates a prediction workload in its domain")


def validate_training(plan, training):
    if plan["model"] != training["model"] or len(training["workloads"]) != 1:
        raise ValueError("structure training needs the task model and one independent workload")
    requests = training["workloads"][0]["requests"]
    if any(row["requests"] == requests for row in plan["workloads"]):
        raise ValueError("the structure training job duplicates a prediction workload")


def main():
    parser = argparse.ArgumentParser(prog="prologue explore", description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    try:
        print(json.dumps(explore(args), indent=2))
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
