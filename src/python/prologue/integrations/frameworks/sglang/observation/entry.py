"""Capture, export or interpret one native job from the worker CLI."""

import argparse
from pathlib import Path

from prologue.common.components import contributions
from prologue.common.io import read_json, write_json
from prologue.integrations.bindings import device_runtime, route

from . import probes
from .probes import ProbeInput

serving_modes = {
    "resource-trace": "observe resources and input reads for independent comparison",
    "service-reference": "observe structure, resources and same-run CPU activity",
    "latency-only": "record scheduler ingress and token publication only",
    "host-timing": "sample native call cuts and CPU-only queue activity",
    "attention": "measure selected native Decode attention calls",
}


def run_capture(args):
    job = read_json(args.input)
    selection = route()
    selection.require_job(job)

    if args.probe is not None:
        args.probe.run(ProbeInput(args.input, args.output, job, selection.revision))
        return

    from .native_session import NativeServingSession

    mode = args.mode
    attention = job.pop("attention_sites") if mode == "attention" else None
    NativeServingSession(job, args.output, mode, args.local_shards, attention).run()
    if attention:

        # Export only after the workers and their observation windows have closed.
        device_runtime().scoped_tasks.publish(args.output)


def sampling_cuts(args):
    from .layout import publish_cuts

    publish_cuts(args.input, args.work, args.rank, args.identities, args.runner, args.output)


def export(args):
    device_runtime().export.run(args.input, args.rank, args.output)


def interpret(args):
    from .layout import serving_facts

    facts = serving_facts(args.input, args.rank, args.trace, evidence=args.evidence)
    write_json(args.output, facts, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="run one normalized native observation job")
    capture.set_defaults(run=run_capture, mode=None, probe=None)
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--local-shards", action="store_true", help="execute logical shards on the single mounted device"
    )
    observation = capture.add_mutually_exclusive_group()

    for mode, description in serving_modes.items():
        observation.add_argument("--" + mode, dest="mode", action="store_const", const=mode, help=description)
    for probe in contributions(probes, "probes"):
        observation.add_argument("--" + probe.name, dest="probe", action="store_const", const=probe,
                                 help=probe.description)

    reader = commands.add_parser("interpret", help="interpret an SDK-exported worker recording")
    reader.set_defaults(run=interpret)
    reader.add_argument("--trace", type=Path, required=True)
    reader.add_argument("--evidence", action="store_true", help="retain measured boundary clocks for source auditing")
    exporter = commands.add_parser("export", help="export one worker through an isolated SDK workspace")
    exporter.set_defaults(run=export)
    cuts = commands.add_parser("sampling-cuts", help="bind source device work to portable native sampling cuts")
    cuts.set_defaults(run=sampling_cuts)

    for option in ("work", "identities", "runner"):
        cuts.add_argument("--" + option, type=Path, required=True)
    for command in (reader, exporter, cuts):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--rank", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    args.run(args)
