"""Recover one prediction or publish its metrics and first concrete evidence gap."""

import json
import subprocess
import time

from prologue.artifacts import write_json
from prologue.serving.latency import RequestLatency
from prologue.workflow.limits import AllowanceExhausted


class ExplorationTarget:
    def __init__(self, task, index, source, configuration):
        self.task = task
        self.root = task / "targets" / str(index)
        self.source = source
        self.identity = {"configuration": configuration, "workload": source["workload"]["id"]}

    def predict(self, predictor, options, release, limit):
        outcome = self.root / "outcome.json"
        if outcome.is_file():
            return json.loads(outcome.read_bytes())
        input_path = self.root / "input.json"
        if not input_path.is_file():
            write_json(input_path, self.source, indent=2, replace=False)
        attempts = self.root / "attempts"
        attempts.mkdir(exist_ok=True)
        previous = sorted(attempts.iterdir(), key=lambda path: int(path.name))
        output = previous[-1] if previous else attempts / "0"

        if previous:
            predictor.recover(output)
        result, evidence = self.evidence(predictor, output)
        elapsed = None
        if result is None:
            if previous:
                output = attempts / str(int(output.name) + 1)
            options.output = output
            start = time.monotonic_ns()
            try:
                predictor(options, release, self.source, limit).run()
                result, evidence = self.evidence(predictor, output)
            except AllowanceExhausted:
                raise
            except (ValueError, RuntimeError, subprocess.SubprocessError, TimeoutError) as failure:
                timed_out = isinstance(failure, (subprocess.TimeoutExpired, TimeoutError))
                result = {"status": "prediction_timeout" if timed_out else "prediction_failed", "message": str(failure)}
            elapsed = time.monotonic_ns() - start

        row = {**self.identity, **result, "directory": str(output.relative_to(self.task)), "elapsed_ns": elapsed}
        if evidence is not None:
            row["evidence"] = str(evidence.relative_to(self.task))
        write_json(outcome, row, indent=2, replace=False)
        return row

    def evidence(self, predictor, output):
        result = output / "result.json"
        if result.is_file():
            metrics = RequestLatency.predicted(json.loads(result.read_bytes()), self.source["workload"]["requests"]).report()
            write_json(output / "request-latency.json", metrics.pop("requests"))
            return {"status": "predicted", **metrics}, result
        if gap := predictor.gap(output):
            status, path = gap
            return {"status": status}, path
        return None, None
