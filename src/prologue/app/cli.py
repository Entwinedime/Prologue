"""Prologue command line entry point."""

import argparse
import json
from pathlib import Path

from prologue.app.acquire import plan_costs, sample_costs
from prologue.app.core import run_core, run_simulation
from prologue.app.releases import (
    control_inputs,
    cost_quotes,
    pinned,
    publish,
    query_structure,
    read_document,
    structure_targets,
)
from prologue.app.task import Plan
from prologue.assets.publication import Store
from prologue.integrations.platforms.ascend.evidence import AscendEvidence, AscendRecording
from prologue.rules.catalog import Rules


def template_document(recording, projections=None):
    if recording.windows is None:
        return recording.read().document()

    bindings = recording.bindings()
    if len(bindings) == 1:
        return recording.read(bindings[0]).document() if projections is None else projections[bindings[0]]
    workers = []
    evidence = {}
    for rank in sorted(recording.profiles):
        observation = recording.read(rank).document() if projections is None else projections[rank]
        workers.append({"rank": rank, "document": {key: observation[key] for key in ("calls", "work")}})
        evidence[rank] = observation["evidence"]
    document = {"domain": recording.domain, "workers": workers, "rank_bindings": bindings,
                "publication_rank": recording.publication_rank(), "evidence": evidence}
    if recording.domain["configuration"].get("pp", 1) > 1:
        from prologue.integrations.frameworks.sglang.observation.pipeline_source import PipelinePrograms

        pipeline = PipelinePrograms(recording, workers, bindings)
        document["programs"] = pipeline.programs()
        for worker in workers:
            worker["document"]["calls"] = pipeline.observed[worker["rank"]].normalize()
    return document


def compile_source(args):
    model_config = None
    if args.pd_scope:
        from prologue.integrations.frameworks.sglang.observation.pd_source import PDSource

        document, source = PDSource(args.observation).document(args.pd_scope)
    else:
        recording = AscendRecording(args.observation)
        if "roles" in recording.manifest:
            document, source, model_config = pd_template_document(recording)
        else:
            document = template_document(recording)
            model_config = recording.manifest.get("model_config")
            source = {"purpose": "template_source", "mode": recording.mode,
                      "origin_raw_ns": recording.delivery["origin_raw_ns"]}
            if recording.windows is not None:
                source["placement"] = recording.manifest["placement"]
                source["observed_ranks"] = sorted(recording.profiles)
                source["rank_bindings"] = recording.bindings()

    if args.model_config:
        supplied = read_document(args.model_config)
        if model_config is not None and model_config != supplied:
            raise ValueError("the supplied model configuration disagrees with the observation")
        model_config = supplied
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("the observation requires its original --model-config")

    rules = Rules.read(args.rules)
    rules.require_observation(source["purpose"], source["mode"])
    if args.for_queries:
        batch = read_document(args.for_queries)
        if batch["domain"] != document["domain"]:
            raise ValueError("source domain does not match the structure queries")
        targets = structure_targets(batch)
        document["required_queries"] = [row["query"] for row in targets]
    asset = run_core("templates", document)
    asset["source"] = source
    asset["model_config"] = model_config
    return asset


def pd_template_document(recording):
    from prologue.integrations.frameworks.sglang.observation.pd_recording import PDRecording

    pair = PDRecording(recording.directory, recording.manifest)
    windows = pair.qualify()
    decode = {rank: pair.qualify_decode(rank, pair.decode.read(rank).document()) for rank in pair.decode.profiles}
    partitions = [
        template_document(pair.prefill),
        template_document(pair.decode, decode),
        pair.transport_document(windows),
    ]
    document = {"domain": pair.domain, "partitions": partitions, "evidence": {
        "windows": len(windows), "requests": len(windows[0]["handoffs"]),
        "admission_sizes": [[len(batch) for batch in window["admissions"]] for window in windows],
    }}
    source = {"purpose": "template_source", "mode": recording.mode, "placement": recording.manifest["placement"],
              "origin_raw_ns": windows[0]["origin_raw_ns"],
              "ranks": {"prefill": pair.prefill.bindings(), "decode": pair.decode.bindings()}}
    return document, source, pair.model_config


def detailed_reference(recording):
    from prologue.verification.structure import StructureReference

    observation = recording.read()
    return StructureReference(observation.document(), observation.publications, observation.manifest)


def structure_reference(recording, rules):
    if "roles" in recording.manifest:
        from prologue.integrations.frameworks.sglang.observation.pd_recording import PDRecording
        from prologue.verification.disaggregation import PDStructureReference

        return PDStructureReference(PDRecording(recording.directory, recording.manifest))
    if recording.windows is None:
        return detailed_reference(recording)
    if recording.domain["configuration"].get("pp", 1) == 1:
        from prologue.verification.parallel import ParallelStructureReference

        return ParallelStructureReference(recording)

    from prologue.app.releases import prepare_templates, query_templates
    from prologue.verification.pipeline import PipelineReference

    reference = PipelineReference(recording)
    document = template_document(recording, {rank: trace.evidence for rank, trace in reference.traces.items()})
    measured_asset = run_core("templates", document)
    measured_asset["model_config"] = recording.manifest["model_config"]
    measured_asset["source"] = {"purpose": "structure_reference"}
    measured_bank = prepare_templates([measured_asset], rules.templates)
    queries = [entry["query"] for entry in measured_asset["roots"]]
    graphs, = query_templates(measured_bank, rules.templates,
                              [{"domain": recording.domain, "queries": queries}], include_graph=True)
    reference.prepare(document["programs"], graphs)
    return reference


def validate_structure_sources(recording, templates):
    if not any(source["domain"] == recording.domain for source in templates["sources"]):
        raise ValueError("template and reference domains disagree")
    if "roles" in recording.manifest:
        delivery = AscendRecording(recording.directory / "decode").delivery
    else:
        delivery = recording.delivery
    origin = delivery["origin_raw_ns"]
    if any(source["source"]["origin_raw_ns"] == origin for source in templates["sources"]):
        raise ValueError("the structure reference was used to build templates")


def structure_job(reference, templates, rules):
    job = {"domain": reference.domain, "workload": reference.workload,
           "templates": templates, **control_inputs(templates, reference.domain, reference.workload),
           "rules": rules.document, "record_structure": True}
    if ingress := getattr(reference, "ingress", None):
        job["measured_ingress"] = ingress
    if getattr(reference, "record_cache", False):
        job["record_cache"] = True
    return job


def verify_structure(args):
    store, release, rules = pinned(args)
    templates = store.asset(release, "templates")
    recording = AscendRecording(args.reference)
    validate_structure_sources(recording, templates)

    reference = structure_reference(recording, rules)
    job = structure_job(reference, templates, rules)
    result = reference.compare(run_simulation(job, args.socket, reference))
    return {**result, "release": release["release"]}


def fit_costs(args):
    from prologue.calibration.costs import Calibration

    query = read_document(args.for_queries) if args.for_queries else None
    calibration = Calibration(Rules.read(args.rules), query)
    for directory in args.device:
        calibration.device(directory)
    for directory in args.collective or ():
        calibration.collective(directory)
    for directory in args.host or ():
        calibration.host(directory)
    return calibration.fit(run_core("coordinates", calibration.batches))


def predict(args):
    store, release, rules = pinned(args)
    templates = store.asset(release, "templates")
    costs = store.asset(release, "performance")
    task = Plan(args.task).document()
    if {"configuration": args.configuration, "workload": args.workload} not in task["targets"]:
        raise ValueError("the requested target is absent from the task")
    configuration = next(row for row in task["configurations"] if row["id"] == args.configuration)
    workload = next(row for row in task["workloads"] if row["id"] == args.workload)
    revisions = {entry["domain"]["framework_revision"] for entry in templates["sources"]
                 if entry["domain"]["model"] == task["model"] and entry["domain"]["configuration"] == configuration}
    if len(revisions) != 1:
        raise ValueError("the target model and configuration require exactly one published framework revision")
    domain = {"framework_revision": revisions.pop(), "model": task["model"], "configuration": configuration}

    job = {"domain": domain, "cost_domain": {**domain, "hardware": read_document(args.hardware)}, "workload": workload,
           "costs": costs, "templates": templates, **control_inputs(templates, domain, workload),
           "rules": rules.document}
    result = run_core("predict", job, str(args.socket))
    return {**result, "domain": domain, "workload": workload, "cost_sources": costs["sources"],
            "release": release["release"]}


def verify_performance(args):
    from prologue.verification.fixed_graph import FixedGraph

    store, release, rules = pinned(args)
    recording = AscendRecording(args.reference)
    reference = detailed_reference(recording)
    costs = store.asset(release, "performance")
    if any(row["origin_raw_ns"] == recording.delivery["origin_raw_ns"] for row in costs["sources"]):
        raise ValueError("the fixed-graph reference was used in calibration")
    graph = FixedGraph(reference, recording.manifest, AscendEvidence.hardware(args.reference))
    baseline = graph.check_baseline(run_core("replay", graph.job()))
    report = {"status": baseline["status"], "baseline": baseline, "release": release["release"]}
    if baseline["status"] != "matched":
        return report

    estimate = graph.compare(run_core("replay", graph.job(costs, rules.document)))
    report.update(status=estimate["status"], estimate=estimate)
    if estimate["status"] == "complete":
        report["restored"] = {kind: graph.compare(run_core("replay", graph.job(costs, rules.document, kind)))
                              for kind in sorted(graph.kinds)}
    return report


def score(args):
    from prologue.verification.autonomous import AutonomousReference

    return AutonomousReference(args.reference).score(read_document(args.prediction))


def quote(args):
    store, release, rules = pinned(args)
    query = read_document(args.query)
    return {"release": release["release"], "quotes": cost_quotes(store.asset(release, "performance"), rules, query)}


def release_arguments(parser):
    parser.add_argument("--store", type=Path, default=Path("data/libraries"))
    parser.add_argument("--release", default="current", help="release UUID or current")


def main() -> None:
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate a task and list its targets")
    plan.add_argument("task", type=Path)
    plan.set_defaults(handler=lambda args: Plan(args.task).document())

    compile_command = commands.add_parser("compile", help="compile hierarchical templates from a detailed observation")
    compile_command.add_argument("observation", type=Path)
    compile_command.add_argument("--rules", type=Path)
    compile_command.add_argument("--model-config", type=Path, help="model configuration if absent from the source")
    compile_command.add_argument("--pd-scope", choices=("copy", "assembly"), help="compile a native PD cut")
    compile_command.add_argument("--for-queries", type=Path, help="limit roots to the original structural queries")
    compile_command.set_defaults(handler=compile_source)

    structure = commands.add_parser("verify-structure", help="compare expanded serving actions to a native reference")
    structure.add_argument("reference", type=Path)
    structure.add_argument("--socket", type=Path, required=True)
    release_arguments(structure)
    structure.set_defaults(handler=verify_structure)

    fit = commands.add_parser("fit-costs", help="fit models from native training observations")
    fit.add_argument("--device", type=Path, action="append", required=True)
    fit.add_argument("--collective", type=Path, action="append", help="include full physical collective timings")
    fit.add_argument("--host", type=Path, action="append")
    fit.add_argument("--rules", type=Path)
    fit.add_argument("--for-queries", type=Path, help="fit only the original work coordinates")
    fit.set_defaults(handler=fit_costs)

    publication = commands.add_parser("publish", help="atomically publish template, performance and rule assets")
    publication.add_argument("--store", type=Path, default=Path("data/libraries"))
    publication.add_argument("--base", default="current", help="expected current release UUID, current or empty")
    libraries = publication.add_mutually_exclusive_group()
    libraries.add_argument("--library", type=Path, action="append", help="complete set of template sources")
    libraries.add_argument("--add-library", type=Path, action="append", help="add to the base template sources")
    publication.add_argument("--require-structure", type=Path, help="require coverage for the original query batch")
    costs = publication.add_mutually_exclusive_group()
    costs.add_argument("--costs", type=Path)
    costs.add_argument("--add-costs", type=Path, help="add uncovered work coordinates to the published costs")
    publication.add_argument("--require-costs", type=Path, help="require coverage for the original cost queries")
    publication.add_argument("--rules", type=Path)
    publication.add_argument("--rule-cases", type=Path, help="native cross-domain witnesses for template rules")
    publication.add_argument("--preview", action="store_true", help="qualify a candidate without changing the release")
    publication.set_defaults(handler=publish)

    inspect = commands.add_parser("show-release", help="show the immutable record of a published release")
    release_arguments(inspect)
    inspect.set_defaults(handler=lambda args: Store(args.store).release(args.release))

    lookup = commands.add_parser("query-structure", help="resolve structural queries and group shared gaps")
    lookup.add_argument("query", type=Path)
    lookup.add_argument("--candidate", type=Path, action="append", help="include an unpublished template source")
    lookup.add_argument("--graph", action="store_true", help="include expanded graphs for covered queries")
    release_arguments(lookup)
    lookup.set_defaults(handler=query_structure)

    query = commands.add_parser("quote-costs", help="query modeled service for bound work coordinates")
    query.add_argument("query", type=Path)
    release_arguments(query)
    query.set_defaults(handler=quote)

    sampling = commands.add_parser("plan-costs", help="group known cost gaps into eligible native sampling jobs")
    sampling.add_argument("query", type=Path)
    release_arguments(sampling)
    sampling.set_defaults(handler=plan_costs)

    collection = commands.add_parser("sample-costs", help="run one bounded sampling job in the native observer image")
    collection.add_argument("plan", type=Path)
    collection.add_argument("output", type=Path)
    collection.add_argument("--job", type=int, default=0)
    collection.add_argument("--attempts", type=int, required=True)
    collection.add_argument("--card-seconds", type=int, required=True)
    collection.add_argument("--max-bytes", type=int, required=True)
    collection.set_defaults(handler=sample_costs)

    forecast = commands.add_parser("predict", help="predict a target using a pinned release")
    forecast.add_argument("task", type=Path)
    forecast.add_argument("--configuration", required=True)
    forecast.add_argument("--workload", required=True)
    forecast.add_argument("--hardware", type=Path, required=True)
    forecast.add_argument("--socket", type=Path, required=True)
    release_arguments(forecast)
    forecast.set_defaults(handler=predict)

    fixed = commands.add_parser("verify-performance", help="compare measured and modeled costs on a fixed native graph")
    fixed.add_argument("reference", type=Path)
    release_arguments(fixed)
    fixed.set_defaults(handler=verify_performance)

    scoring = commands.add_parser("score", help="score a prediction against an independent lightweight recording")
    scoring.add_argument("prediction", type=Path)
    scoring.add_argument("reference", type=Path)
    scoring.set_defaults(handler=score)
    args = parser.parse_args()

    try:
        output = args.handler(args)
        compact = args.command in ("compile", "fit-costs", "quote-costs")
        print(json.dumps(output, separators=(",", ":"), indent=None if compact else 2))
    except (KeyError, OSError, TypeError, ValueError, StopIteration) as error:
        parser.error(str(error))
