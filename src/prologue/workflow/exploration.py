"""Explore planned targets from a common release and retain each observed gap."""

import json
import subprocess

from prologue.acquisition.planning import cost_demands
from prologue.workflow.budget import Acquisition, acquisition_usage


def load_demands(forecasts):
    structure = {}
    costs = []
    for target, row in forecasts.items():
        gap = row.get("gap")
        if gap is None:
            continue
        if "coordinates" in gap:
            costs.append({"target": target, "domain": gap["domain"], "coordinates": gap["coordinates"]})
        else:
            domain = gap.get("domain", row["domain"])
            key = json.dumps(domain, sort_keys=True)
            group = structure.setdefault(key, {"domain": domain, "targets": []})
            group["targets"].append({"target": target, **gap})
    return {"structure": list(structure.values()), "costs": costs}


class Exploration:
    def __init__(self, output, plan, initial, deployment, supplement=None):
        self.output = output
        self.targets = {f"{index:04}": target for index, target in enumerate(plan["targets"], 1)}
        self.initial = initial
        self.deployment = deployment
        self.supplement = supplement

    def predict(self, target, release, directory):
        path = directory / "prediction.json"
        if path.exists():
            return Acquisition.read(path)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            report = self.deployment.predict(self.targets[target], release, directory)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            report = {"status": "prediction_failed", "reason": str(error)}
        if report.get("release", release) != release:
            raise ValueError("prediction used a different release")
        report["release"] = release
        Acquisition.write(path, report)
        return report

    def run(self):
        release = self.initial
        affected = self.targets
        forecasts = {}
        locations = {}
        publications = []
        round_number = 0
        while True:
            round_number += 1
            directory = self.output / "rounds" / f"{round_number:04}"
            for target in affected:
                case = directory / target
                forecasts[target] = self.predict(target, release, case)
                locations[target] = str((case / "prediction.json").relative_to(self.output))

            gaps = load_demands(forecasts)
            Acquisition.write(directory / "demands.json", {"structure": gaps["structure"],
                                                           "costs": cost_demands(gaps["costs"])})
            if all(row["status"] == "complete" for row in forecasts.values()):
                stop = "complete"
                break
            if self.supplement is None:
                stop = "readonly"
                break

            journal = directory / "supplement.json"
            if journal.exists():
                result = Acquisition.read(journal)
            else:
                try:
                    result = self.supplement.run(release, gaps, directory)
                except (TimeoutError, subprocess.TimeoutExpired) as error:
                    result = {"status": "processing_budget_exhausted", "reason": str(error)}
                Acquisition.write(journal, result)
            if result["status"] != "published":
                stop = result["status"]
                break
            if result["release"] == release or not result["affected"]:
                raise ValueError("supplementation must advance a release for an affected target")
            release = result["release"]
            affected = result["affected"]
            publications.append(result["publication"])

        usage = acquisition_usage(self.output / "acquisition")
        usage["processing_ns"] = self.deployment.clock.charged_ns
        report = {
            "status": "complete" if stop == "complete" else "partial",
            "stop": stop,
            "initial_release": self.initial,
            "final_release": release,
            "rounds": round_number,
            "usage": usage,
            "publications": publications,
            "targets": [{"target": name, **target, "status": forecasts[name]["status"],
                         "release": forecasts[name]["release"], "result": locations[name]}
                        for name, target in self.targets.items()],
        }
        Acquisition.write(self.output / "result.json", report)
        return report
