"""Prologue command line entry point."""

import argparse
import json
from pathlib import Path

from prologue.app.core import run_core, run_simulation
from prologue.app.task import Plan
from prologue.integrations.platforms.ascend.evidence import AscendEvidence


def main() -> None:
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate a task and list its targets")
    plan.add_argument("task", type=Path, help="path to a task JSON file")
    compile_command = commands.add_parser("compile", help="compile hierarchical templates from a detailed observation")
    compile_command.add_argument("observation", type=Path, help="path to a completed observation directory")
    structure = commands.add_parser("verify-structure", help="compare native control and expanded templates to a reference")
    structure.add_argument("reference", type=Path)
    structure.add_argument("--library", type=Path, action="append", required=True)
    structure.add_argument("--model-config", type=Path, required=True)
    structure.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "plan":
            print(json.dumps(Plan(args.task).document(), indent=2))
        elif args.command == "compile":
            evidence = AscendEvidence.open(args.observation).document()
            print(json.dumps(run_core("templates", evidence), separators=(",", ":")))
        else:
            from prologue.verification.structure import StructureReference

            observation = AscendEvidence.open(args.reference)
            reference = StructureReference(observation.document(), observation.publications, observation.manifest)
            libraries = [json.loads(path.read_text(encoding="utf-8")) for path in args.library]
            model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
            prediction = run_simulation(reference.job(libraries, model_config), args.socket, reference)
            print(json.dumps(reference.compare(prediction), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
