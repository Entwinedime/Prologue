"""Prologue command line entry point."""

import argparse
import json
from pathlib import Path

from prologue.app.core import run_core, run_simulation
from prologue.app.task import Plan
from prologue.integrations.platforms.ascend.evidence import AscendEvidence


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def template_sources(sources, domain):
    libraries = []
    for source in sources:
        if source["domain"] != domain:
            raise ValueError("template source and target have different domains")
        libraries.append({"definitions": source["definitions"],
                          "instances": [row for row in source["instances"] if row["parent"] is None]})
    return libraries


def detailed_reference(path):
    from prologue.verification.structure import StructureReference

    observation = AscendEvidence.open(path)
    return observation, StructureReference(observation.document(), observation.publications, observation.manifest)


def verify_structure(args):
    _, reference = detailed_reference(args.reference)
    sources = [read(path) for path in args.library]
    job = {"domain": reference.domain, "workload": reference.workload,
           "libraries": template_sources(sources, reference.domain), "model_config": read(args.model_config),
           "record_structure": True}
    return reference.compare(run_simulation(job, args.socket, reference))


def fit_costs(args):
    from prologue.calibration.costs import Calibration

    calibration = Calibration()
    for directory in args.device:
        calibration.device(directory)
    for directory in args.host:
        calibration.host(directory)
    return calibration.fit(run_core("coordinates", calibration.batches))


def predict(args):
    task = Plan(args.task).document()
    if {"configuration": args.configuration, "workload": args.workload} not in task["targets"]:
        raise ValueError("the requested target is absent from the task")
    configuration = next(row for row in task["configurations"] if row["id"] == args.configuration)
    workload = next(row for row in task["workloads"] if row["id"] == args.workload)
    sources = [read(path) for path in args.library]
    domain = {"framework_revision": sources[0]["domain"]["framework_revision"],
              "model": task["model"], "configuration": configuration}
    costs = read(args.costs)
    job = {"domain": domain, "cost_domain": {**domain, "hardware": read(args.hardware)}, "workload": workload,
           "costs": costs, "libraries": template_sources(sources, domain), "model_config": read(args.model_config)}
    result = run_core("predict", job, str(args.socket))
    return {**result, "domain": domain, "workload": workload, "cost_sources": costs["sources"]}


def verify_performance(args):
    from prologue.verification.fixed_graph import FixedGraph

    observation, reference = detailed_reference(args.reference)
    costs = read(args.costs)
    if any(row["origin_raw_ns"] == observation.manifest["delivery"]["origin_raw_ns"] for row in costs["sources"]):
        raise ValueError("the fixed-graph reference was used in calibration")
    graph = FixedGraph(reference, observation.manifest, observation.hardware(args.reference))
    baseline = graph.check_baseline(run_core("replay", graph.job()))
    if baseline["status"] != "matched":
        return {"status": baseline["status"], "baseline": baseline}

    estimate = graph.compare(run_core("replay", graph.job(costs)))
    report = {"status": estimate["status"], "baseline": baseline, "estimate": estimate}
    if estimate["status"] == "complete":
        report["restored"] = {kind: graph.compare(run_core("replay", graph.job(costs, kind)))
                              for kind in sorted(graph.kinds)}
    return report


def score(args):
    from prologue.verification.autonomous import AutonomousReference

    return AutonomousReference(args.reference).score(read(args.prediction))


def quote(args):
    query = read(args.query)
    return run_core("quote", {**query, "costs": read(args.costs)})


def control_arguments(parser):
    parser.add_argument("--library", type=Path, action="append", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate a task and list its targets")
    plan.add_argument("task", type=Path)
    plan.set_defaults(handler=lambda args: Plan(args.task).document())

    compile_command = commands.add_parser("compile", help="compile hierarchical templates from a detailed observation")
    compile_command.add_argument("observation", type=Path)
    compile_command.set_defaults(handler=lambda args: run_core("templates", AscendEvidence.open(args.observation).document()))

    structure = commands.add_parser("verify-structure", help="compare native control and expanded templates to a reference")
    structure.add_argument("reference", type=Path)
    control_arguments(structure)
    structure.set_defaults(handler=verify_structure)

    fit = commands.add_parser("fit-costs", help="fit models from native training observations")
    fit.add_argument("--device", type=Path, action="append", required=True)
    fit.add_argument("--host", type=Path, action="append", required=True)
    fit.set_defaults(handler=fit_costs)

    query = commands.add_parser("quote-costs", help="query modeled service for bound work coordinates")
    query.add_argument("query", type=Path)
    query.add_argument("--costs", type=Path, required=True)
    query.set_defaults(handler=quote)

    forecast = commands.add_parser("predict", help="predict a target with explicit template and cost assets")
    forecast.add_argument("task", type=Path)
    forecast.add_argument("--configuration", required=True)
    forecast.add_argument("--workload", required=True)
    forecast.add_argument("--costs", type=Path, required=True)
    forecast.add_argument("--hardware", type=Path, required=True)
    control_arguments(forecast)
    forecast.set_defaults(handler=predict)

    fixed = commands.add_parser("verify-performance", help="compare measured and modeled durations on a fixed native graph")
    fixed.add_argument("reference", type=Path)
    fixed.add_argument("--costs", type=Path, required=True)
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
