"""Record a planned target using an installed native SGLang runtime."""

import argparse
import json
from pathlib import Path

from prologue.integrations.frameworks.sglang.observation.session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--mode", choices=("detailed", "light", "host-timing"), required=True)
    args = parser.parse_args()

    receipt = json.loads(args.plan.read_text(encoding="utf-8"))
    wanted = {"configuration": args.configuration, "workload": args.workload}
    if wanted not in receipt["targets"]:
        parser.error("the requested configuration and workload are not a planned target")
    Session(receipt, args.configuration, args.workload, args.output, args.mode).run()


if __name__ == "__main__":
    main()
