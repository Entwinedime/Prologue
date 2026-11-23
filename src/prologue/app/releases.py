"""Assemble and qualify the assets used by public commands."""

import json

from prologue.app.core import run_core
from prologue.assets.publication import Store
from prologue.rules.catalog import Rules


def read_document(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pinned(args):
    store = Store(args.store)
    release = store.release(args.release)
    rules = Rules(store.asset(release, "rules"))
    return store, release, rules


def prepare_templates(paths, model_config):
    sources = [read_document(path) for path in paths]
    domain = sources[0]["domain"]
    libraries = []
    for source in sources:
        if source["domain"] != domain:
            raise ValueError("template sources must belong to the same domain")
        roots = []
        for row in source["instances"]:
            if row["parent"] is None:
                roots.append({
                    "parent": None, "parameters": {"phase": row["parameters"]["phase"]},
                    "members": [{"kv_committed_tokens": member["kv_committed_tokens"]} for member in row["members"]],
                    "definition": row["definition"],
                })
        libraries.append({"definitions": source["definitions"], "instances": roots, "source": source["source"]})
    return {"domain": domain, "libraries": libraries, "model_config": model_config}


def qualify(documents, rules):
    templates = documents["templates"]
    costs = documents["performance"]
    domain = {key: value for key, value in costs["domain"].items() if key != "hardware"}
    if domain != templates["domain"]:
        raise ValueError("template and performance asset domains disagree")
    for library in templates["libraries"]:
        rules.require_observation("template_source", library["source"]["mode"])
    for source in costs["sources"]:
        rules.require_observation(source["family"], source["mode"])
    run_core("qualify-release", {"libraries": templates["libraries"], "costs": costs,
                                "rules": rules.document})


def publish(args):
    store = Store(args.store)
    base = args.base
    if base == "current":
        base = store.current()
    elif base == "empty":
        base = None

    snapshot = store.snapshot(base)
    documents = snapshot.documents.copy()
    previous = documents.get("templates")
    model_config = previous["model_config"] if previous else None
    if args.model_config:
        model_config = read_document(args.model_config)
    if args.library:
        if model_config is None:
            raise ValueError("the initial template asset requires --model-config")
        templates = prepare_templates(args.library, model_config)
        if previous and templates["domain"]["model"] != previous["domain"]["model"] and not args.model_config:
            raise ValueError("changing the model requires --model-config")
        documents["templates"] = templates
    elif args.model_config and previous:
        documents["templates"] = {**previous, "model_config": model_config}
    if args.costs:
        documents["performance"] = read_document(args.costs)
    if "templates" not in documents or "performance" not in documents:
        raise ValueError("the initial publication requires --library, --model-config and --costs")

    if args.rules or "rules" not in documents:
        rules = Rules.read(args.rules)
        documents["rules"] = rules.document
    else:
        rules = Rules(documents["rules"])

    qualify(documents, rules)
    return store.publish(snapshot, documents)
