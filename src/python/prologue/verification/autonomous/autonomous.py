"""Verify cost-driven serving against an independent lightweight native run."""

from subprocess import TimeoutExpired

from prologue.common.io import read_json
from prologue.common.observation import CapturedWorkers
from prologue.host.libraries.release import LibraryRelease
from prologue.host.workflow.options import PredictionOptions
from prologue.host.workflow.prediction import PredictionProcess
from prologue.verification.runner import VerificationRun

from .latency_metrics import LatencyComparison, NativeLatencyMetrics


class AutonomousVerification(VerificationRun):
    def __init__(self, arguments, adapter):
        super().__init__(arguments)
        self.adapter = adapter
        self.report = {
            "input": str(arguments.input.resolve()),
            "native_task": str(arguments.capture.resolve()),
            "libraries": str(arguments.libraries.resolve()),
        }

    def qualify_reference(self, recording, source):
        recorded = recording.inputs
        if (not recorded.get("latency_only") or recorded.get("host_timing") or "attention_sites" in recorded["job"]
                or not all("latency" in instance for instance in recording.manifest["instances"])):
            return "reference_requires_latency_capture"
        if self.input_mismatch(source):
            return "input_mismatch"

        instances = recording.manifest["instances"]
        if not CapturedWorkers(instances).joint_execution:
            return "reference_requires_joint_execution"
        sessions = recording.sessions(recording.source, instances)

        if self.input_mismatch(source, sessions=sessions):
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
                status, path = process.gap()
                self.report["evidence"] = str(path)
                return status
        except (RuntimeError, TimeoutError, TimeoutExpired) as error:
            self.report["prediction_error"] = str(error)
            return "prediction_failed"

        self.report["prediction_result"] = str(output / "result.json")
        return "completed"

    def verify(self):
        args = self.arguments
        source = read_json(args.input)
        release = LibraryRelease(args.libraries)
        options = PredictionOptions(output=self.output / "prediction", timeout=args.timeout, binary=args.binary,
                                  modeling_image=args.modeling_image, control_image=args.control_image,
                                  execution_inputs=None, execution_trace=args.details)
        process = PredictionProcess(options, release, source, self.adapter)

        if process.task.binding is None:
            self.report["gap"] = process.task.missing_workers()
            return self.finish("template_gap")
        recording = self.recording

        if status := self.training_gap(templates=process.task.binding, costs=release.costs):
            return self.finish(status)

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
