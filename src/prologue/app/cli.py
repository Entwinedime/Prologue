"""Prologue command line entry point."""

import argparse
import json
from pathlib import Path

from prologue.app.task import Plan


def main() -> None:
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate a task and list its targets")
    plan.add_argument("task", type=Path, help="path to a task JSON file")
    args = parser.parse_args()

    try:
        result = Plan(args.task).document()
    except (OSError, TypeError, ValueError) as error:
        plan.error(str(error))
    print(json.dumps(result, indent=2))
