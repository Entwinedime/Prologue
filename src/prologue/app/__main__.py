"""Parse public task commands and select their concrete application handlers."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict", help="predict one target from a fixed template and cost release")
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--timeout", type=float, required=True)
    prediction.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    prediction.add_argument("--execution-trace", action="store_true", help="stream optional expanded graphs, dependencies and completion events")
    prediction.add_argument("--execution-inputs", type=Path, help="use checked per-actor time conditions and native ingress rounds")
    production = commands.add_parser("produce", help="project a closed native capture into graphs or source templates")
    production.add_argument("--product", choices=("graphs", "templates"), default="templates")
    production.add_argument("--capture", type=Path, required=True, help="published native capture task directory")
    production.add_argument("--budget-seconds", type=float, required=True, help="cumulative allowance across resumes; may be explicitly raised")
    production.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    production.add_argument("--keep-intermediates", action="store_true")
    capture = commands.add_parser("capture", help="capture or resume a native task with cumulative allowances")
    capture.add_argument("--input", type=Path, required=True)
    local = commands.add_parser("acquire-local", help="answer compatible prediction gaps in one bounded native session")
    local.add_argument("--query", type=Path, nargs="+", action="extend", required=True, help="original structure or work queries")
    local.add_argument("--libraries", type=Path, help="fixed release providing native cuts for stateful work")
    local.add_argument("--structure-session", type=Path, action="append", default=[],
                       help="qualified training session whose native cut may be timed independently")
    local.add_argument("--binary", type=Path, help="Docker-built template compiler for qualifying scheduler action queries")
    local.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    local.add_argument("--storage-bytes", type=int, help="optional storage allowance under the session directory")
    publish = commands.add_parser("publish-local", help="publish qualified local evidence to a persistent library store")
    publish.add_argument("--libraries", type=Path, required=True)
    publish.add_argument("--session", type=Path, required=True)
    publish.add_argument("--query", type=Path, nargs="+", action="extend", required=True)
    publish.add_argument("--timeout", type=float, required=True, help="time allowance for offline compilation and query checks")
    for command in (capture, local):
        command.add_argument("--deployment", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--timeout", type=float, required=True)
        command.add_argument("--budget-seconds", type=float, required=True, help="cumulative native allowance across attempts and resumes")
        command.add_argument("--attempt-limit", type=int, required=True, help="cumulative native attempts; each invocation starts at most one")
        command.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    observation = capture.add_mutually_exclusive_group()
    observation.add_argument("--resource-trace", action="store_true", help="record optional resource and input-read evidence")
    observation.add_argument("--service-reference", action="store_true", help="record detailed structure and same-run CPU activity")
    observation.add_argument("--latency-only", action="store_true", help="run without the profiler and record scheduler latency")
    observation.add_argument("--host-timing", action="store_true", help="record native CPU work and queue cuts")
    observation.add_argument("--attention-sites", type=Path, help="measure named Decode modules in a pooled serving comparison")
    samples = commands.add_parser("cost-samples", help="publish measured services from a projected source")
    samples.add_argument("--production", type=Path, required=True, help="published training production that owns the observation")
    samples.add_argument("--projection", type=Path, help="regenerated graph and facts for the same native observation")
    samples.add_argument("--rank", type=int, action="append", required=True, help="repeat with matching work and output paths")
    samples.add_argument("--work-identities", type=Path, action="append", required=True)
    samples.add_argument("--transport-only", action="store_true", help="export only independently measured peer services and physical placement")
    samples.add_argument("--output", type=Path, action="append", required=True)
    batch_samples = commands.add_parser("batch-cost-samples", help="extract measured work from a qualified local batch session")
    batch_samples.add_argument("--session", type=Path, required=True)
    batch_samples.add_argument("--timeout", type=float, required=True, help="modeling allowance in seconds")
    hosts = commands.add_parser("host-cost-samples", help="qualify independently measured native host services")
    for name in ("detailed", "host", "facts", "trace", "graph", "work-identities", "output"):
        hosts.add_argument("--" + name, type=Path, required=True)
    costs = commands.add_parser("fit-costs", help="fit registered cost models from qualified service samples")
    costs.add_argument("--samples", type=Path, nargs="+", required=True)
    costs.add_argument("--check", type=Path, action="append", default=[])
    costs.add_argument("--demand", type=Path, nargs=2, action="append", default=[], metavar=("GRAPH", "WORK"))
    costs.add_argument("--gap", type=Path, action="append", default=[], help="recheck a prior prediction's actual cost demand")
    costs.add_argument("--output", type=Path, required=True)
    costs.add_argument("--binary", type=Path, help="Docker-built prologue binary for demand and held-out sample checks")
    costs.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    costs.add_argument("--timeout", type=float, default=300, help="cumulative modeling allowance for cost checks in seconds")
    sampling = commands.add_parser("acquire-costs", help="fulfil known cost requirements with bounded native sampling")
    sampling.add_argument("--production", type=Path, action="append", required=True,
                          help="training sources for native cuts and samples; supply demand when no --demand or --gap is given")
    sampling.add_argument("--samples", type=Path, nargs="+", default=[], help="previously qualified training samples")
    sampling.add_argument("--keep-intermediates", action="store_true", help="retain reconstructed projections and work identities after fitting")
    sampling.add_argument("--demand", type=Path, nargs=2, action="append", default=[], metavar=("GRAPH", "WORK"))
    sampling.add_argument("--gap", type=Path, action="append", default=[], help="actual cost demand from a prior prediction")
    exploration = commands.add_parser("explore", help="predict a configuration/workload set from one fixed release")
    exploration.add_argument("--input", type=Path, required=True)
    exploration.add_argument("--output", type=Path, help="task directory, defaults to a new UTC-stamped data directory")
    exploration.add_argument("--details", action="store_true", help="emit target execution traces")
    exploration.add_argument("--timeout", type=float, required=True, help="maximum seconds per prediction")
    exploration.add_argument("--prediction-workers", type=int, default=1, help="concurrent device-free target predictions")
    exploration.add_argument("--storage-bytes", type=int, help="task output allowance in fixed mode, or acquisition files and new library bytes with --deployment")
    exploration.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    exploration.add_argument("--deployment", type=Path, help="collect missing evidence locally at this physical placement")
    exploration.add_argument("--card-budget-seconds", type=float, help="cumulative time across physical acquisition devices")
    exploration.add_argument("--postprocessing-budget-seconds", type=float, help="cumulative local qualification and publication time")
    exploration.add_argument("--attempt-limit", type=int, help="cumulative physical acquisition attempts across targets")
    exploration.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    for command in (prediction, exploration):
        command.add_argument("--libraries", type=Path, required=True, help="fixed template and cost release directory")
    sampling.add_argument("--transport-samples", type=Path, nargs="+", default=[], help="external peer service profiles for the target placement")
    sampling.add_argument("--timeout", type=float, required=True, help="maximum seconds per native attempt")
    sampling.add_argument("--native-budget-seconds", type=float, required=True)
    sampling.add_argument("--postprocessing-budget-seconds", type=float, required=True)
    sampling.add_argument("--attempt-limit", type=int, required=True)
    sampling.add_argument("--storage-bytes", type=int, required=True, help="monitored logical bytes under this collection root")
    sampling.add_argument("--source-image", default="prologue-sglang:ubuntu22.04")
    for command in (production, prediction, sampling, batch_samples):
        command.add_argument("--output", type=Path, required=True)
    for command in (production, prediction, sampling, exploration, batch_samples, publish):
        command.add_argument("--binary", type=Path, required=True)
        command.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    arguments = parser.parse_args()
    if arguments.command in {"capture", "produce", "acquire-costs", "acquire-local", "publish-local"}:
        from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
    if arguments.command in {"fit-costs", "acquire-costs"}:
        from prologue.integrations.frameworks.sglang.acquisition.cost_reuse import BatchHostReuse
    if arguments.command == "produce":
        from prologue.acquisition.production import CaptureProduction
        from prologue.acquisition.recording import PublishedCapture

        CaptureProduction(arguments, PublishedCapture(arguments.capture), SGLangAscendLaunch).run()
    elif arguments.command == "acquire-local":
        from prologue.libraries import LibraryRelease

        queries = [json.loads(path.read_bytes()) for path in arguments.query]
        release = LibraryRelease(arguments.libraries) if arguments.libraries else None
        result = SGLangAscendLaunch.local_acquisition(release, arguments).run(arguments, queries, json.loads(arguments.deployment.read_bytes()))
        raise SystemExit(0 if result["status"] == "qualified" else 1)
    elif arguments.command == "publish-local":
        from prologue.acquisition.library_increment import LocalLibraryIncrement
        from prologue.acquisition.library_store import LibraryStore
        from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples
        from prologue.integrations.runtimes.docker import ModelingProcess

        model = ModelingProcess(arguments.binary, arguments.modeling_image, arguments.timeout)
        acquisition = SGLangAscendLaunch.local_acquisition(None, arguments)
        queries = [json.loads(path.read_bytes()) for path in arguments.query]
        increment = LocalLibraryIncrement(arguments.session, queries, acquisition, model, CostSamples)
        published = LibraryStore(arguments.libraries).publish(increment)
        print(json.dumps({"release": published.identity, "directory": str(published.directory)}))
    elif arguments.command == "capture":
        sites = json.loads(arguments.attention_sites.read_bytes()) if arguments.attention_sites else None
        launch = SGLangAscendLaunch(json.loads(arguments.input.read_text()), json.loads(arguments.deployment.read_text()))
        launch.capture(arguments, attention_sites=sites, latency_only=arguments.latency_only, host_timing=arguments.host_timing,
                       service_reference=arguments.service_reference).run()
    elif arguments.command == "cost-samples":
        from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples

        CostSamples.export(arguments)
    elif arguments.command == "batch-cost-samples":
        from prologue.artifacts import write_json
        from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples
        from prologue.integrations.runtimes.docker import ModelingProcess

        model = ModelingProcess(arguments.binary, arguments.modeling_image, arguments.timeout)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(arguments.output, CostSamples.batch(arguments.session, model, arguments.output.parent), replace=False)
    elif arguments.command == "host-cost-samples":
        from prologue.integrations.frameworks.sglang.ascend.host_samples import HostCostSamples

        HostCostSamples(arguments).run()
    elif arguments.command == "fit-costs":
        from prologue.acquisition.fitting import FitCosts

        FitCosts(arguments, BatchHostReuse.match).run()
    elif arguments.command == "acquire-costs":
        from prologue.acquisition.costs import CostAcquisition
        from prologue.acquisition.fitting import TransportSamples

        profiles = TransportSamples(arguments.transport_samples)
        CostAcquisition(arguments, SGLangAscendLaunch, BatchHostReuse.match, transport_samples=profiles).run()
    elif arguments.command == "explore":
        from prologue.app.prediction import PredictionProcess
        from prologue.workflow.exploration import ExplorationTask

        task = ExplorationTask(arguments, PredictionProcess)
        acquisition = None
        if arguments.deployment:
            from prologue.acquisition.replenishment import Replenishment
            from prologue.integrations.frameworks.sglang.ascend.device_samples import CostSamples
            from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch

            acquisition = Replenishment(task, arguments, SGLangAscendLaunch.local_acquisition, CostSamples)
        elif any(x is not None for x in (arguments.card_budget_seconds, arguments.postprocessing_budget_seconds, arguments.attempt_limit)):
            parser.error("acquisition allowances require --deployment")
        if not task.run(acquisition):
            raise SystemExit(1)
    else:
        from prologue.app.prediction import PredictionProcess
        from prologue.libraries import LibraryRelease

        process = PredictionProcess(arguments, LibraryRelease(arguments.libraries), json.loads(arguments.input.read_bytes()))
        raise SystemExit(0 if process.run() else 1)


if __name__ == "__main__":
    main()
