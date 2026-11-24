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

    def predict(self, predictor, options, release, limit, basis):
        outcome = self.root / "outcome.json"
        if outcome.is_file():
            row = json.loads(outcome.read_bytes())
            if row["basis"] == basis:
                return row
        input_path = self.root / "input.json"
        if not input_path.is_file():
            write_json(input_path, self.source, indent=2, replace=False)
        journal = self.root / "attempt.json"
        previous = json.loads(journal.read_bytes()) if journal.is_file() else None
        result, evidence, elapsed = None, None, None
        if previous:
            output = self.root / "attempts" / str(previous["number"])
            predictor.recover(output)
            if previous["basis"] == basis:
                result, evidence = self.evidence(predictor, output)
        if result is None:
            number = previous["number"] + 1 if previous else 0
            output = self.root / "attempts" / str(number)
            previous = {"number": number, "basis": basis, "release": release.identity}
            write_json(journal, previous)
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

        row = {**self.identity, **result, "basis": basis, "release": previous["release"],
               "directory": str(output.relative_to(self.task)), "elapsed_ns": elapsed}
        if evidence is not None:
            row["evidence"] = str(evidence.relative_to(self.task))
        write_json(outcome, row, indent=2)
        return row

    def evidence(self, predictor, output):
        result = output / "result.json"
        if result.is_file():
            metrics = RequestLatency.predicted(json.loads(result.read_bytes()), self.source["workload"]["requests"]).report()
            write_json(output / "request-latency.json", metrics.pop("requests"))
            return {"status": "predicted", **metrics}, result
        if gap := predictor.gap(output, self.source):
            status, path = gap
            return {"status": status}, path
        return None, None
