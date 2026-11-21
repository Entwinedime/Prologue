"""Run one target against published assets and retain the first concrete gap."""

import json
import subprocess
import time
from types import SimpleNamespace

from prologue.artifacts import write_json
from prologue.serving.latency import RequestLatency
from prologue.workflow.limits import AllowanceExhausted


class ExplorationAsset:
    def __init__(self, sources, costs):
        self.sources, self.costs = sources, costs.resolve(strict=True)
        self.production = sources.directory

    def document(self):
        return {"production": str(self.production), "costs": str(self.costs)}


class ExplorationTarget:
    def __init__(self, root, index, model, configuration, workload, predictor):
        self.task = root
        self.predictor = predictor
        self.root = root / "targets" / str(index)
        self.input = {"model": model, "instances": configuration["instances"], "workload": workload}
        self.identity = {"configuration": configuration["id"], "workload": workload["id"]}

    def reuse(self, assets, runtime, limit):
        best_gap = None
        eligible = (asset for asset in assets if asset.sources.select(self.input) is not None)
        for index, asset in enumerate(eligible):
            stage = "reused" if index == 0 else f"reused-{index}"
            row = self.predict(runtime, limit, asset, stage)
            if row["status"] not in {"template_gap", "cost_gap"}:
                return asset, row

            # A cost gap has an instantiated template. It can be completed without structural recapture.
            if best_gap is None or (best_gap[1]["status"] == "template_gap" and row["status"] == "cost_gap"):
                best_gap = asset, row
        return best_gap

    def predict(self, arguments, limit, asset, stage):
        input_path = self.root / "input.json"
        if not input_path.is_file():
            write_json(input_path, self.input, indent=2, replace=False)

        directory = self.root / stage
        outcome = directory / "outcome.json"
        if outcome.is_file():
            row = json.loads(outcome.read_text())
            if row["asset"] != asset.document():
                raise ValueError("target outcome belongs to different assets")
            return row
        directory.mkdir(parents=True, exist_ok=True)
        attempts = sorted((path for path in directory.iterdir() if path.is_dir()), key=lambda path: int(path.name))
        output = attempts[-1] if attempts else directory / "0"

        # A stopped coordinator can leave two named processes or a completed publication.
        if attempts and not (output / "result.json").is_file():
            self.predictor.recover(output)
        result, evidence = self.evidence(output)
        elapsed = None
        if result is None:
            if attempts:
                output = directory / str(int(output.name) + 1)
            start = time.monotonic_ns()
            try:
                self.predictor(SimpleNamespace(
                    **arguments, input=input_path, output=output, production=asset.production,
                    costs=asset.costs, execution_inputs=None,
                ), limit).run(source=self.input, productions=asset.sources)
            except AllowanceExhausted:
                raise
            except (ValueError, RuntimeError, subprocess.SubprocessError, TimeoutError) as failure:
                result, evidence = self.evidence(output)
                if result is None:
                    timed_out = isinstance(failure, (subprocess.TimeoutExpired, TimeoutError))
                    result = {"status": "prediction_timeout" if timed_out else "prediction_failed", "message": str(failure)}
            else:
                result, evidence = self.evidence(output)
            elapsed = time.monotonic_ns() - start

        row = {**self.identity, **result, "asset": asset.document(), "asset_use": stage,
               "directory": str(output.relative_to(self.task)), "elapsed_ns": elapsed}
        if evidence is not None:
            row["evidence"] = str(evidence.relative_to(self.task))
        write_json(outcome, row, indent=2, replace=False)
        return row

    def evidence(self, output):
        result = output / "result.json"
        if result.is_file():
            metrics = RequestLatency.predicted(json.loads(result.read_text()), self.input["workload"]["requests"]).report()
            requests = metrics.pop("requests")
            write_json(output / "request-latency.json", requests)
            return {"status": "predicted", **metrics}, result
        for status, name in (("template_gap", "template-gap.json"), ("cost_gap", "cost-gap.json")):
            path = output / name
            if path.is_file():
                return {"status": status}, path
        return None, None
