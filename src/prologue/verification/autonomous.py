"""Verify cost-driven serving against an independent lightweight native run."""

import argparse
import json
from pathlib import Path
from subprocess import TimeoutExpired

from prologue.acquisition.observation.workers import CapturedWorkers
from prologue.app.prediction import PredictionProcess
from prologue.libraries.release import LibraryRelease
from prologue.runtime.io import write_json
from prologue.workflow.options import PredictionOptions

from .latency_metrics import LatencyComparison, NativeLatencyMetrics
from .reference import ReferenceRecording


class AutonomousVerification:
    def __init__(self, arguments):
        self.arguments = arguments
        self.output = arguments.output.resolve()
        self.report = {
            "input": str(arguments.input.resolve()),
            "native_task": str(arguments.capture.resolve()),
            "libraries": str(arguments.libraries.resolve()),
        }

    def finish(self, status):
        write_json(self.output / "result.json", {"status": status, **self.report}, indent=2, replace=False)
        return status == "compared"

    def qualify_reference(self, recording, source):
        recorded = recording.inputs
        if (not recorded.get("latency_only") or recorded.get("host_timing") or "attention_sites" in recorded["job"]
                or not all("latency" in instance for instance in recording.manifest["instances"])):
            return "reference_requires_latency_capture"
        if difference := recording.input_difference(recorded["job"], source):
            self.report["input_difference"] = difference
            return "input_mismatch"

        instances = recording.manifest["instances"]
        if not CapturedWorkers(instances).joint_execution:
            return "reference_requires_joint_execution"
        sessions = recording.sessions(recording.source, instances)
        if difference := recording.session_difference(source, instances, sessions):
            self.report["input_difference"] = difference
            return "input_mismatch"

        native = NativeLatencyMetrics(recording.source, instances, sessions).report()
        self.report["native"] = native
        if native["specialization_loads"] is None:
            return "reference_missing_specialization_evidence"
        if native["specialization_loads"]:
            return "reference_loaded_specializations"
        rows = native["requests"]
        if any(actual["declared_ns"] != expected["arrival_ns"]
               for actual, expected in zip(rows, source["workload"]["requests"], strict=True)):
            return "reference_changed_arrivals"

        submissions = ["submitted_ns"]
        if "decode_submitted_ns" in rows[0]:
            submissions.append("decode_submitted_ns")
        order = sorted(rows, key=lambda row: row["declared_ns"])
        self.report["max_submission_delay_ns"] = max(
            row[field] - row["declared_ns"] for field in submissions for row in rows
        )
        if any(order != sorted(rows, key=lambda row: row[field]) for field in submissions):
            return "reference_reordered_submissions"
        return "qualified"

    def predict(self, process):
        output = process.arguments.output
        try:
            if not process.run():
                status, path = process.gap(output)
                self.report["evidence"] = str(path)
                return status
        except (RuntimeError, TimeoutError, TimeoutExpired) as error:
            self.report["prediction_error"] = str(error)
            return "prediction_failed"
        self.report["prediction_result"] = str(output / "result.json")
        return "completed"

    def run(self):
        args = self.arguments
        self.output.mkdir(parents=True)
        source = json.loads(args.input.read_text())
        release = LibraryRelease(args.libraries)
        options = PredictionOptions(output=self.output / "prediction", timeout=args.timeout, binary=args.binary,
                                  modeling_image=args.modeling_image, control_image=args.control_image,
                                  execution_inputs=None, execution_trace=args.details)
        process = PredictionProcess(options, release, source)
        if process.task.binding is None:
            self.report["gap"] = process.task.missing_workers()
            return self.finish("template_gap")
        recording = ReferenceRecording(args.capture)
        training = self.report["training"] = recording.training(templates=process.task.binding, costs=release.costs)
        if training["status"] != "separate":
            return self.finish(training["status"])

        # Only the scoring step consumes native timing. A cold reference still leaves a real prediction outcome.
        prediction = self.report["prediction_status"] = self.predict(process)
        reference = self.report["reference_status"] = self.qualify_reference(recording, source)
        if prediction != "completed":
            return self.finish(prediction)
        if reference != "qualified":
            return self.finish(reference)

        self.report["prediction"] = LatencyComparison(
            self.report["native"], self.output / "prediction/result.json"
        ).report()
        return self.finish("compared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="completed lightweight native capture task")
    for name in ("libraries", "input", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True, help="prediction allowance in seconds")
    parser.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    parser.add_argument("--details", action="store_true", help="retain the execution trace")
    raise SystemExit(0 if AutonomousVerification(parser.parse_args()).run() else 1)


if __name__ == "__main__":
    main()
