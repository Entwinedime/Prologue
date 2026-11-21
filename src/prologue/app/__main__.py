"""Parse public task commands and select their concrete application handlers."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan-sources", help="select and publish an initial capture input per source domain")
    planning.add_argument("--input", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict", help="predict from published template productions")
    prediction.add_argument("--production", type=Path, required=True, help="published finite source collection; target input selects a complete worker source")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--timeout", type=float, required=True)
    prediction.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    prediction.add_argument("--execution-trace", action="store_true", help="stream optional expanded graphs, dependencies and completion events")
    timing = prediction.add_mutually_exclusive_group()
    timing.add_argument("--execution-inputs", type=Path, help="supply checked per-actor time conditions and native ingress rounds")
    timing.add_argument("--costs", type=Path, help="predict service durations from an independent cost asset")
    production = commands.add_parser("produce", help="project a closed native capture into graphs or source templates")
    production.add_argument("--product", choices=("graphs", "templates"), default="templates")
    production.add_argument("--capture", type=Path, required=True, help="published native capture task directory")
    production.add_argument("--budget-seconds", type=float, required=True, help="cumulative allowance across resumes; may be explicitly raised")
    production.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    production.add_argument("--keep-intermediates", action="store_true")
    capture = commands.add_parser("capture", help="capture or resume a native task with cumulative allowances")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--deployment", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--timeout", type=float, required=True)
    capture.add_argument("--budget-seconds", type=float, required=True, help="cumulative native allowance across attempts and resumes")
    capture.add_argument("--attempt-limit", type=int, required=True, help="cumulative native attempts; each invocation starts at most one")
    capture.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    observation = capture.add_mutually_exclusive_group()
    observation.add_argument("--resource-trace", action="store_true", help="record optional resource and input-read evidence")
    observation.add_argument("--service-reference", action="store_true", help="record detailed structure and same-run CPU activity")
    observation.add_argument("--latency-only", action="store_true", help="run without the profiler and record scheduler latency")
    observation.add_argument("--host-timing", action="store_true", help="record native CPU work and queue cuts")
    samples = commands.add_parser("cost-samples", help="publish measured services from a projected source")
    samples.add_argument("--production", type=Path, required=True, help="published training production that owns the observation")
    samples.add_argument("--projection", type=Path, help="regenerated graph and facts for the same native observation")
    samples.add_argument("--rank", type=int, required=True)
    samples.add_argument("--work-identities", type=Path, required=True)
    samples.add_argument("--transport-only", action="store_true", help="export only independently measured peer services and physical placement")
    samples.add_argument("--output", type=Path, required=True)
    hosts = commands.add_parser("host-cost-samples", help="qualify independently measured native host services")
    for name in ("detailed", "host", "facts", "trace", "graph", "work-identities", "output"):
        hosts.add_argument("--" + name, type=Path, required=True)
    costs = commands.add_parser("fit-costs", help="fit bounded curves from qualified service samples")
    costs.add_argument("--samples", type=Path, nargs="+", required=True)
    costs.add_argument("--check", type=Path, action="append", default=[])
    costs.add_argument("--demand", type=Path, nargs=2, action="append", default=[], metavar=("GRAPH", "WORK"))
    costs.add_argument("--gap", type=Path, action="append", default=[], help="recheck a prior prediction's actual cost demand")
    costs.add_argument("--output", type=Path, required=True)
    costs.add_argument("--binary", type=Path, help="Docker-built prologue binary for demand and held-out sample checks")
    costs.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    costs.add_argument("--timeout", type=float, default=300, help="cumulative modeling allowance for cost checks in seconds")
    sampling = commands.add_parser("acquire-costs", help="fulfil known cost requirements with bounded native sampling")
    sampling.add_argument("--production", type=Path, action="append", required=True, help="training source collection with intermediates")
    sampling.add_argument("--samples", type=Path, nargs="+", default=[], help="previously qualified training samples")
    sampling.add_argument("--demand", type=Path, nargs=2, action="append", default=[], metavar=("GRAPH", "WORK"))
    sampling.add_argument("--gap", type=Path, action="append", default=[], help="actual cost demand from a prior prediction")
    acquisition = commands.add_parser("acquire-sources", help="capture and compile a finite plan under shared cumulative allowances")
    acquisition.add_argument("--plan", type=Path, required=True)
    acquisition.add_argument("--observations", type=Path, help="JSON map of planned source job IDs to published capture task paths; native inputs must match")
    acquisition.add_argument("--deployment", type=Path, required=True)
    acquisition.add_argument("--keep-intermediates", action="store_true")
    acquisition.add_argument("--resource-trace", action="store_true", help="record optional resource and input-read evidence")
    exploration = commands.add_parser("explore", help="prepare assets and predict a configuration/workload set")
    exploration.add_argument("--input", type=Path, required=True)
    exploration.add_argument("--deployment", type=Path, required=True)
    exploration.add_argument("--output", type=Path, help="task directory, defaults to a new UTC-stamped data directory")
    exploration.add_argument("--reuse", type=Path, help="published prior exploration task")
    exploration.add_argument("--details", action="store_true", help="retain source intermediates and emit target execution traces")
    exploration.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    for command in (sampling, exploration):
        command.add_argument("--transport-samples", type=Path, nargs="+", default=[], help="external peer service profiles for the target placement")
    for command in (acquisition, sampling, exploration):
        command.add_argument("--timeout", type=float, required=True, help="maximum seconds per native attempt")
        command.add_argument("--native-budget-seconds", type=float, required=True)
        command.add_argument("--postprocessing-budget-seconds", type=float, required=True)
        command.add_argument("--attempt-limit", type=int, required=True)
        command.add_argument("--storage-bytes", type=int, required=True, help="monitored logical bytes under this collection root")
        command.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    for command in (production, prediction, acquisition, sampling):
        command.add_argument("--output", type=Path, required=True)
    for command in (production, prediction, acquisition, sampling, exploration):
        command.add_argument("--binary", type=Path, required=True)
        command.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    arguments = parser.parse_args()
    if arguments.command in {"capture", "acquire-sources", "acquire-costs", "explore"}:
        from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
    if arguments.command in {"fit-costs", "acquire-costs", "explore"}:
        from prologue.integrations.frameworks.sglang.acquisition.cost_reuse import BatchHostReuse
    if arguments.command == "plan-sources":
        from prologue.acquisition.planning import PlanningProcess
        from prologue.integrations.frameworks.sglang.acquisition.planning import SourcePlanning

        PlanningProcess(SourcePlanning(json.loads(arguments.input.read_text())), arguments.output).run()
    elif arguments.command == "produce":
        from prologue.acquisition.production import CaptureProduction
        from prologue.acquisition.recording import PublishedCapture

        CaptureProduction(arguments, PublishedCapture(arguments.capture)).run()
    elif arguments.command == "capture":
        from prologue.acquisition.capture import CaptureProcess

        launch = SGLangAscendLaunch(json.loads(arguments.input.read_text()), json.loads(arguments.deployment.read_text()))
        CaptureProcess(arguments, launch, latency_only=arguments.latency_only, host_timing=arguments.host_timing,
                       service_reference=arguments.service_reference).run()
    elif arguments.command == "cost-samples":
        from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples

        CostSamples(arguments).run()
    elif arguments.command == "host-cost-samples":
        from prologue.integrations.frameworks.sglang.ascend.host_samples import HostCostSamples

        HostCostSamples(arguments).run()
    elif arguments.command == "fit-costs":
        from prologue.acquisition.fitting import FitCosts

        FitCosts(arguments, BatchHostReuse.match).run()
    elif arguments.command == "acquire-sources":
        from prologue.acquisition.sources import SourceAcquisition

        SourceAcquisition(arguments, SGLangAscendLaunch).run()
    elif arguments.command == "acquire-costs":
        from prologue.acquisition.costs import CostAcquisition
        from prologue.acquisition.fitting import TransportSamples

        profiles = TransportSamples(arguments.transport_samples)
        CostAcquisition(arguments, SGLangAscendLaunch, BatchHostReuse.match, transport_samples=profiles).run()
    elif arguments.command == "explore":
        from prologue.app.prediction import PredictionProcess
        from prologue.workflow.exploration import ExplorationTask

        task = ExplorationTask(arguments, planner=SGLangAscendLaunch.source_planner, place=SGLangAscendLaunch,
                               reuse_samples=BatchHostReuse.match, predictor=PredictionProcess)
        if not task.run():
            raise SystemExit(1)
    else:
        from prologue.app.prediction import PredictionProcess

        PredictionProcess(arguments).run()


if __name__ == "__main__":
    main()
