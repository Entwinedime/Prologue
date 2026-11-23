"""Assemble and qualify the assets used by public commands."""

import json

from prologue.acquisition.planning import structure_demands
from prologue.app.core import run_core
from prologue.assets.publication import Store
from prologue.integrations.frameworks.sglang.configuration import Configuration
from prologue.rules.catalog import Rules
from prologue.rules.review import RuleReview


def read_document(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pinned(args):
    store = Store(args.store)
    release = store.release(args.release)
    rules = Rules(store.asset(release, "rules"))
    return store, release, rules


def prepare_templates(sources, rule, previous=None):
    return run_core("template-bank", {"previous": previous, "sources": sources, "rule": rule})


def template_model(templates, model):
    for entry in templates["models"]:
        if entry["model"] == model:
            return entry["model_config"]
    raise ValueError("the template asset has no configuration for the model")


def control_inputs(templates, domain, workload):
    result = {"model_config": template_model(templates, domain["model"])}
    routes = Configuration(**domain["configuration"]).route_table(workload)
    if routes is None:
        return result

    # Routed layers are part of the published source contract, independent of the reference run.
    layouts = {tuple(row["layer"] for row in root["query"]["expert_routes"])
               for source in templates["sources"] if source["domain"] == domain
               for root in source["roots"] if "expert_routes" in root["query"]}
    if len(layouts) != 1:
        raise ValueError("the template sources require one native routed layer layout for this domain")
    layers, = layouts
    result["expert_routes"] = {"layers": list(layers), "routes": routes, "requests": workload["requests"]}
    return result


def structure_targets(batch):
    targets = batch["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("a structure query requires at least one target")
    for row in targets:
        if (not isinstance(row, dict) or row.keys() != {"target", "query"}
                or not isinstance(row["target"], str) or not row["target"]
                or not isinstance(row["query"], dict) or not row["query"]):
            raise ValueError("each target must have a nonempty name and query object")
    return targets


def query_templates(templates, rule, batches, include_graph=False):
    return run_core("query-structure", {
        "bank": templates, "rule": rule, "batches": batches, "include_graph": include_graph,
    })


def examine_structure(templates, rules, batch, include_graph=False):
    targets = structure_targets(batch)
    queries = [{"domain": batch["domain"], "queries": [row["query"] for row in targets]}]
    decisions, = query_templates(templates, rules.templates, queries, include_graph)
    return structure_demands(targets, decisions)


def query_structure(args):
    store, release, rules = pinned(args)
    templates = store.asset(release, "templates")
    if args.candidate:
        sources = [read_document(path) for path in args.candidate]
        templates = prepare_templates(sources, rules.templates, templates)
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
    if not any(entry["domain"] == domain for entry in templates["sources"]):
        raise ValueError("the performance asset has no corresponding template domain")
    for entry in templates["sources"]:
        source = entry["source"]
        rules.require_observation(source["purpose"], source["mode"])
    for source in costs["sources"]:
        rules.require_observation(source["family"], source["mode"])
    run_core("qualify-release", {"bank": templates, "costs": costs, "rules": rules.document})


def publish(args):
    store = Store(args.store)
    base = args.base
    if base == "current":
        base = store.current()
    elif base == "empty":
        base = None

    snapshot = store.snapshot(base)
    documents = snapshot.documents.copy()
    if args.rules or "rules" not in documents:
        rules = Rules.read(args.rules)
    else:
        rules = Rules(documents["rules"])

    # A full source replacement qualifies sharing from the isolated rule on those same sources.
    previous = None if args.library else documents.get("templates")
    prior_rule = (snapshot.documents["rules"]["templates"] if previous is not None
                  else {**rules.templates, "shared_operations": []})
    changed_rule = prior_rule != rules.templates
    if changed_rule and args.rule_cases is None:
        raise ValueError("changing template sharing requires --rule-cases")

    sources = [read_document(path) for path in (args.library or args.add_library or ())]
    structure_requirements = read_document(args.require_structure) if args.require_structure else None
    if args.add_library:
        if previous is None or structure_requirements is None:
            raise ValueError("adding a template source requires a published base and --require-structure")
        if any(source["domain"] != structure_requirements["domain"] for source in sources):
            raise ValueError("added template sources must match the original query domain")
    if sources or changed_rule:
        documents["templates"] = prepare_templates(sources, rules.templates, previous)

    cost_requirements = read_document(args.require_costs) if args.require_costs else None
    if args.costs:
        documents["performance"] = read_document(args.costs)
    elif args.add_costs:
        if "performance" not in documents or cost_requirements is None:
            raise ValueError("adding costs requires a published base and the original --require-costs query")
        addition = read_document(args.add_costs)
        documents["performance"] = extend_costs(documents["performance"], addition, cost_requirements)
    if "templates" not in documents or "performance" not in documents:
        raise ValueError("the initial publication requires --library and --costs")
    documents["rules"] = rules.document

    if structure_requirements is not None:
        report = examine_structure(documents["templates"], rules, structure_requirements)
        if report["demands"]:
            return {"status": "unresolved_structure", "release": base, **report}
    if cost_requirements is not None:
        quotes = cost_quotes(documents["performance"], rules, cost_requirements)
        missing = [row for row in quotes if row["status"] != "covered"]
        if missing:
            return {"status": "unresolved_costs", "release": base,
                    "domain": cost_requirements["domain"], "missing": missing}
    qualify(documents, rules)

    review = None
    if args.rule_cases:
        candidate = documents["templates"]
        prior = prepare_templates(sources, prior_rule, previous) if sources else previous
        comparison = RuleReview(candidate, read_document(args.rule_cases))
        before = query_templates(prior, prior_rule, comparison.batches, include_graph=True)
        after = query_templates(candidate, rules.templates, comparison.batches, include_graph=True)
        review = comparison.compare(before, after, prior_rule, rules.templates)
        if not review["passed"]:
            return {"status": "rule_cases_failed", "release": base, "rule_review": review}

    if args.preview:
        return {"status": "qualified", "release": base, "rule_review": review}
    return store.publish(snapshot, documents, review)
