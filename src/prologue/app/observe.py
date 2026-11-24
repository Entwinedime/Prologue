"""Record a planned target using an installed native SGLang runtime."""

import argparse
import json
from pathlib import Path

from prologue.integrations.frameworks.sglang.configuration import Disaggregation, Model, configuration_from
from prologue.integrations.frameworks.sglang.observation.session import Session
from prologue.workflow.budget import Acquisition


def observe(plan, target, output, mode, placement):
    if target not in plan["targets"]:
        raise ValueError("the requested configuration and workload are not a planned target")
    model = Model(**plan["model"])
    configured = next(row for row in plan["configurations"] if row["id"] == target["configuration"])
    configuration = configuration_from(configured)
    workload = next(row for row in plan["workloads"] if row["id"] == target["workload"])
    if isinstance(configuration, Disaggregation):
        from prologue.integrations.frameworks.sglang.observation.disaggregation import DisaggregatedSession

        session = DisaggregatedSession(model, configuration, workload, output, mode, placement)
    else:
        session = Session(model, configuration, workload, output, mode, placement)
    session.run()
    return 2 if isinstance(configuration, Disaggregation) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--configuration")
    parser.add_argument("--workload")
    parser.add_argument("--mode", choices=("detailed", "light", "host-timing"))
    parser.add_argument("--placement", choices=("single_device", "full"), default="single_device")
    parser.add_argument("--attempt", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.attempt:
        directory = args.output / "observation"
        engines = observe(plan["plan"], plan["target"], directory, "detailed", "single_device")
        Acquisition.write(args.output / "native.json", {"status": "observed", "engine_starts": engines,
                                                        "observation": directory.name})
        return
    if not (args.configuration and args.workload and args.mode):
        parser.error("direct observation requires a configuration, workload and mode")
    observe(plan, {"configuration": args.configuration, "workload": args.workload},
            args.output, args.mode, args.placement)


if __name__ == "__main__":
    main()
