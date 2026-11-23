"""Prologue command line entry point."""

import argparse
import json
from pathlib import Path

from prologue.app.core import run_core
from prologue.app.task import Plan
from prologue.integrations.platforms.ascend.evidence import AscendEvidence


def main() -> None:
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate a task and list its targets")
    plan.add_argument("task", type=Path, help="path to a task JSON file")
    compile_command = commands.add_parser("compile", help="compile hierarchical templates from a detailed observation")
    compile_command.add_argument("observation", type=Path, help="path to a completed observation directory")
    args = parser.parse_args()

    try:
        if args.command == "plan":
            print(json.dumps(Plan(args.task).document(), indent=2))
        else:
            evidence = AscendEvidence.read(args.observation)
            print(json.dumps(run_core("templates", evidence), separators=(",", ":")))
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
