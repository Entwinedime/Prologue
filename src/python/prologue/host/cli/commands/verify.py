"""Verify native evidence, execution structure, fixed-graph costs and serving latency."""

from pathlib import Path

from prologue.host.cli.arguments import command, control, modeling, paths, timeout
from prologue.host.libraries.release import LibraryRelease
from prologue.verification.autonomous.autonomous import AutonomousVerification
from prologue.verification.autonomous.latency_metrics import compare_publications
from prologue.verification.autonomous.schedule import compare_schedule
from prologue.verification.core.native_case import NativeCase
from prologue.verification.core.source_evidence import ProductionEvidence
from prologue.verification.fixed.fixed_graph import FixedPerformance
from prologue.verification.structure.compare_actions import ActionComparison
from prologue.verification.structure.execution_inputs import NativeInputs
from prologue.verification.structure.local_graphs import LocalComparison
from prologue.verification.structure.structure import StructureVerification


def register(group):
    native = command(group, "source", "check native evidence and calibrated clocks", source)
    paths(native, "production", "capture", "output")
    native.add_argument("--libraries", type=Path, help="check the published templates against the recording")
    details(native)
    timing = command(group, "inputs", "extract measured conditions for structure verification", inputs)
    paths(timing, "capture", "graphs", "output")
    timing.add_argument("--layouts", type=Path, help="require the measured endpoints for each action")
    local = command(group, "local-graphs", "compare expanded local calls against recorded native actions", local_graphs)
    paths(local, "capture", "graphs", "expanded", "output")

    structure = command(
        group, "structure", "verify execution structure under measured timing", configure=structure_mode
    )
    structure.add_argument("--compare-only", action="store_true", help="compare an existing execution trace")
    fixed = command(group, "fixed-graph", "measure cost errors on a fixed native execution graph", fixed_graph)
    paths(fixed, "capture", "reference", "costs", "output")
    modeling(fixed)
    timeout(fixed)
    details(fixed)
    fixed.add_argument("--work-groups", type=Path, help="replace only named work groups with predicted durations")
    autonomous = command(group, "autonomous", "compare cost-driven serving with a native recording",
                         configure=autonomous_mode)
    mode = autonomous.add_mutually_exclusive_group()
    mode.add_argument("--metrics-only", action="store_true", help="score existing token publications")
    mode.add_argument("--schedule-diff", action="store_true", help="locate the first batch schedule difference")


def details(parser):
    parser.add_argument("--details", action="store_true", help="retain detailed verification evidence")


def structure_mode(parser, args):
    paths(parser, "capture", "output")
    details(parser)
    parser.add_argument("--peer-handoffs", action="store_true", help="check measured single-device handoff ports")
    if args.compare_only:
        paths(parser, "graphs", "execution")
        return compare_actions
    else:
        paths(parser, "reference", "libraries", "input")
        modeling(parser)
        control(parser)
        timeout(parser)
        return verify_structure


def autonomous_mode(parser, args):
    paths(parser, "output")
    if args.metrics_only:
        parser.add_argument("--capture", type=Path, nargs="+", required=True)
        parser.add_argument("--prediction", type=Path, nargs="+", help="prediction result files in capture order")
        return latency_metrics
    elif args.schedule_diff:
        paths(parser, "capture", "graphs", "execution")
        return schedule
    else:
        paths(parser, "capture", "libraries", "input")
        modeling(parser)
        control(parser)
        timeout(parser)
        details(parser)
        return verify_autonomous


def source(args):
    check = ProductionEvidence(args.production, args.capture, args.output, args.details)
    _, document = check.inspect()
    if args.libraries is not None:
        document["published_templates"] = check.published_templates(LibraryRelease(args.libraries))
    return check.publish(document)


def inputs(args):
    report = NativeInputs(args.capture, args.graphs, args.output, args.layouts).run()
    return 0 if report["status"] == "extracted" else 1


def local_graphs(args):
    return LocalComparison(args.capture, args.graphs, args.expanded, args.output).run()


def compare_actions(args):
    case = NativeCase.read(args.capture, args.graphs)
    report = ActionComparison(case, args.execution, args.output, args.details, peer_handoffs=args.peer_handoffs).run()
    return 0 if report["status"] == "actions_matched" else 1


def verify_structure(args):
    return 0 if StructureVerification(args, args.adapters.prediction).run() else 1


def fixed_graph(args):
    return 0 if FixedPerformance(args).run() else 1


def latency_metrics(args):
    compare_publications(args.capture, args.prediction, args.output)


def schedule(args):
    compare_schedule(args.capture, args.graphs, args.execution, args.output)


def verify_autonomous(args):
    return 0 if AutonomousVerification(args, args.adapters.prediction).run() else 1
