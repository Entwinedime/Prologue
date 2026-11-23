"""Assemble and qualify the assets used by public commands."""

import json

from prologue.acquisition.planning import structure_demands
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


def prepare_templates(paths, model_config, previous=None):
    sources = [read_document(path) for path in paths]
    domain = previous["domain"] if previous else sources[0]["domain"]
    libraries = previous["libraries"].copy() if previous else []
    for source in sources:
        if source["domain"] != domain:
            raise ValueError("template sources must belong to the same domain")
        library = {key: source[key] for key in ("definitions", "roots", "source")}
        if library not in libraries:
            libraries.append(library)
    return {"domain": domain, "libraries": libraries, "model_config": model_config}


def structure_targets(domain, batch):
    if batch["domain"] != domain:
        raise ValueError("source domain does not match the structure queries")
    targets = batch["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("a structure query requires at least one target")
    for row in targets:
        if (not isinstance(row, dict) or row.keys() != {"target", "query"}
                or not isinstance(row["target"], str) or not row["target"]
                or not isinstance(row["query"], dict) or not row["query"]):
            raise ValueError("each target must have a nonempty name and query object")
    return targets


def examine_structure(templates, rules, batch, include_graph=False):
    targets = structure_targets(templates["domain"], batch)
    decisions = run_core("query-structure", {
        "libraries": templates["libraries"], "rule": rules.templates,
        "queries": [row["query"] for row in targets], "include_graph": include_graph,
    })
    return structure_demands(targets, decisions)


def query_structure(args):
    store, release, rules = pinned(args)
    templates = store.asset(release, "templates")
    if args.candidate:
        templates = prepare_templates(args.candidate, templates["model_config"], templates)
    batch = read_document(args.query)
    return {"release": release["release"], "preview": bool(args.candidate),
            **examine_structure(templates, rules, batch, args.graph)}


def cost_quotes(costs, rules, query):
    return run_core("quote", {**query, "costs": costs, "rules": rules.document})


def extend_costs(previous, addition, required):
    if previous["domain"] != addition["domain"]:
        raise ValueError("additional costs require the same physical domain")
    models = {json.dumps(row["coordinate"], sort_keys=True): row for row in previous["models"]}
    wanted = {json.dumps(row, sort_keys=True) for row in required["coordinates"]}
    count = len(models)
    for row in addition["models"]:
        key = json.dumps(row["coordinate"], sort_keys=True)
        if key in wanted:
            models.setdefault(key, row)
    if len(models) == count:
        return previous

    sources = previous["sources"] + [row for row in addition["sources"] if row not in previous["sources"]]
    return {**previous, "sources": sources, "models": list(models.values())}


def qualify(documents, rules):
    templates = documents["templates"]
    costs = documents["performance"]
    domain = {key: value for key, value in costs["domain"].items() if key != "hardware"}
    if domain != templates["domain"]:
        raise ValueError("template and performance asset domains disagree")
    for library in templates["libraries"]:
        source = library["source"]
        rules.require_observation(source["purpose"], source["mode"])
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
    cost_requirements = read_document(args.require_costs) if args.require_costs else None
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
    elif args.add_library:
        if previous is None:
            raise ValueError("adding a template source requires a published base")
        if not args.require_structure:
            raise ValueError("adding a template source requires the original query batch with --require-structure")
        documents["templates"] = prepare_templates(args.add_library, model_config, previous)
    elif args.model_config and previous:
        documents["templates"] = {**previous, "model_config": model_config}
    if args.costs:
        documents["performance"] = read_document(args.costs)
    elif args.add_costs:
        if "performance" not in documents or cost_requirements is None:
            raise ValueError("adding costs requires a published base and the original --require-costs query")
        addition = read_document(args.add_costs)
        documents["performance"] = extend_costs(documents["performance"], addition, cost_requirements)
    if "templates" not in documents or "performance" not in documents:
        raise ValueError("the initial publication requires --library, --model-config and --costs")

    if args.rules or "rules" not in documents:
        rules = Rules.read(args.rules)
        documents["rules"] = rules.document
    else:
        rules = Rules(documents["rules"])

    if args.require_structure:
        batch = read_document(args.require_structure)
        report = examine_structure(documents["templates"], rules, batch)
        if report["demands"]:
            return {"status": "unresolved_structure", "release": base, **report}
    if cost_requirements is not None:
        quotes = cost_quotes(documents["performance"], rules, cost_requirements)
        missing = [row for row in quotes if row["status"] != "covered"]
        if missing:
            return {"status": "unresolved_costs", "release": base,
                    "domain": cost_requirements["domain"], "missing": missing}
    qualify(documents, rules)
    return store.publish(snapshot, documents)
