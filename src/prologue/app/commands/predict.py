"""Run one prediction or compare a set of targets."""

from pathlib import Path

from prologue.acquisition.options import ReplenishmentOptions
from prologue.acquisition.replenishment import Replenishment
from prologue.app.arguments import command, control, modeling, options, paths, source_image, timeout
from prologue.app.registry import AdapterRegistry
from prologue.libraries.release import LibraryRelease
from prologue.runtime.io import read_json
from prologue.workflow.exploration import ExplorationTask
from prologue.workflow.options import ExplorationOptions, PredictionOptions
from prologue.workflow.prediction import PredictionProcess


def register(group):
    single = command(group, "run", "predict one target from a fixed release", run)
    matrix = command(group, "explore", "compare configurations and workloads from a fixed release", explore)
    for parser in (single, matrix):
        paths(parser, "input", "libraries")
        timeout(parser)
        modeling(parser)
        control(parser)
    paths(single, "output")
    single.add_argument("--execution-trace", action="store_true", help="retain expanded graphs and timeline events")
    single.add_argument("--execution-inputs", type=Path, help="use measured execution conditions")
    matrix.add_argument("--output", type=Path, help="task directory, defaults to a new UTC-stamped directory")
    matrix.add_argument("--details", action="store_true", help="retain per-target execution traces")
    matrix.add_argument("--prediction-workers", type=int, default=1)
    matrix.add_argument("--storage-bytes", type=int)
    matrix.add_argument("--deployment", type=Path, help="enable bounded acquisition at this placement")
    matrix.add_argument("--card-budget-seconds", type=float)
    matrix.add_argument("--postprocessing-budget-seconds", type=float)
    matrix.add_argument("--attempt-limit", type=int)
    source_image(matrix)


def run(args):
    process = PredictionProcess(options(PredictionOptions, args), LibraryRelease(args.libraries),
                                read_json(args.input), AdapterRegistry.select().prediction)
    return 0 if process.run() else 1


def explore(args):
    source = read_json(args.input)
    framework = source["framework"]
    adapters = AdapterRegistry.select(framework)
    task = ExplorationTask(options(ExplorationOptions, args), adapters.prediction, framework, source)
    acquisition = None
    if args.deployment:
        acquisition = Replenishment(task, options(ReplenishmentOptions, args), adapters.native.local_acquisition,
                                    adapters.samples)
    elif any(x is not None for x in (args.card_budget_seconds, args.postprocessing_budget_seconds, args.attempt_limit)):
        args.parser.error("acquisition allowances require --deployment")
    return 0 if task.run(acquisition) else 1
