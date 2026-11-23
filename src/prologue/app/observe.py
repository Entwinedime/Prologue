"""Record a planned target using an installed native SGLang runtime."""

import argparse
import json
from pathlib import Path

from prologue.integrations.frameworks.sglang.configuration import Disaggregation, Model, configuration_from
from prologue.integrations.frameworks.sglang.observation.session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--mode", choices=("detailed", "light", "host-timing"), required=True)
    parser.add_argument("--placement", choices=("single_device", "full"), default="single_device")
    args = parser.parse_args()

    receipt = json.loads(args.plan.read_text(encoding="utf-8"))
    wanted = {"configuration": args.configuration, "workload": args.workload}
    if wanted not in receipt["targets"]:
        parser.error("the requested configuration and workload are not a planned target")
    model = Model(**receipt["model"])
    configured = next(row for row in receipt["configurations"] if row["id"] == args.configuration)
    configuration = configuration_from(configured)
    workload = next(row for row in receipt["workloads"] if row["id"] == args.workload)
    if isinstance(configuration, Disaggregation):
        from prologue.integrations.frameworks.sglang.observation.disaggregation import DisaggregatedSession

        session = DisaggregatedSession(model, configuration, workload, args.output, args.mode, args.placement)
    else:
        session = Session(model, configuration, workload, args.output, args.mode, args.placement)
    session.run()


if __name__ == "__main__":
    main()
