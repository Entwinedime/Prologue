"""Verify a serving structure against one observed native case."""

from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from tempfile import TemporaryDirectory
from time import monotonic_ns

from prologue.common.io import read_json
from prologue.host.libraries.release import LibraryRelease
from prologue.host.runtime.docker import ModelingProcess
from prologue.host.workflow.options import PredictionOptions
from prologue.host.workflow.prediction import PredictionProcess
from prologue.verification.core.source import ProductionEvidence
from prologue.verification.runner import VerificationRun

from .compare_actions import ActionComparison
from .execution_inputs import NativeInputs


class StructureVerification(VerificationRun):
    success = {"full_structure_matched", "structure_matched_within_scope"}

    def __init__(self, arguments, adapter):
        super().__init__(arguments)
        self.adapter = adapter

    def verify(self):
        args = self.arguments
        recording = self.recording
        capture = recording.source
        wall_ns = {}
        stages = self.report = {
            "capture": str(recording.directory),
            "reference": str(args.reference.resolve(strict=True)),
            "libraries": str(args.libraries.resolve()),
            "wall_ns": wall_ns,
        }
        source = read_json(args.input)

        if self.input_mismatch(source):
            return self.finish("input_mismatch")
        prediction = self.output / "prediction"
        timing = self.output / "timing"
        options = PredictionOptions(
            output=prediction,
            timeout=args.timeout,
            binary=args.binary,
            modeling_image=args.modeling_image,
            control_image=args.control_image,
            execution_inputs=timing / "execution-inputs.json",
            execution_trace=True,
        )
        process = PredictionProcess(options, LibraryRelease(args.libraries), source, self.adapter)
        binding = process.task.binding

        if binding is None:
            return self.finish("template_gap", gap=process.task.missing_workers())
        if args.peer_handoffs and binding.worker_deployment.get("acquisition_mode", "full") != "single_device":
            return self.finish("peer_handoff_source_unqualified")
        if status := self.training_gap(templates=binding):
            return self.finish(status)

        started = monotonic_ns()
        checker = ProductionEvidence(
            args.reference, recording.raw_source, self.output / "source-evidence", args.details
        )
        reference, evidence = checker.inspect()
        stages["reference_status"] = evidence["native_case"]["status"]
        status = checker.publish(evidence, require_joint=True)
        wall_ns["reference"] = monotonic_ns() - started

        if status:
            return self.finish("reference_unqualified")

        if self.input_mismatch(source, case=reference):
            return self.finish("input_mismatch")

        inputs = self.extract_timing(capture, reference, timing)
        if isinstance(inputs, str):
            return self.finish(inputs)

        return self.check_prediction(process, prediction, reference, inputs)

    def check_prediction(self, process, prediction, reference, inputs):
        args = self.arguments
        stages, wall_ns = self.report, self.report["wall_ns"]
        started = monotonic_ns()
        try:
            if not process.run():
                stages["prediction_error"] = "published library has an uncovered target operation"
        except (RuntimeError, TimeoutError, TimeoutExpired) as error:
            stages["prediction_error"] = str(error)

        wall_ns["prediction"] = monotonic_ns() - started
        trace = prediction / ("execution.partial.jsonl" if "prediction_error" in stages else "execution.jsonl")
        if not trace.is_file():
            if gap := process.gap():
                status, path = gap
                return self.finish(status, evidence=str(path))

            return self.finish("prediction_no_trace")

        # Compare against the original reference, including actual condition consumption and independent causal
        # evidence.
        started = monotonic_ns()
        try:
            report = ActionComparison(
                reference,
                trace,
                self.output / "comparison",
                args.details,
                realization=inputs.realization,
                peer_handoffs=args.peer_handoffs,
            ).run()
        except ValueError as error:
            wall_ns["comparison"] = monotonic_ns() - started
            return self.finish("comparison_unverified", comparison_error=str(error))

        wall_ns["comparison"] = monotonic_ns() - started
        stages["comparison_status"] = report["status"]
        stages["coverage"] = report["coverage"]
        stages["actions_compared"] = report["actions_compared"]

        if "prediction_error" in stages:
            if rejected := report["first_rejected_action"]:
                stages["first_rejected_action"] = rejected
                return self.finish("measured_layout_difference")

            if gap := process.gap():
                status, path = gap
                return self.finish(status, evidence=str(path))
            return self.finish("prediction_failed")

        if report["status"] != "actions_matched":
            return self.finish(report["status"])
        if report["condition_inputs"]["full_node_coverage"]:
            return self.finish("full_structure_matched")

        return self.finish("structure_matched_within_scope")

    def extract_timing(self, capture, reference, timing) -> NativeInputs | str:
        args = self.arguments
        started = monotonic_ns()
        with TemporaryDirectory(prefix=".action-layouts-", dir=self.output) as temporary:
            layouts = Path(temporary)
            modeling = ModelingProcess(args.binary, args.modeling_image, args.timeout)
            try:
                for rank in reference.sources:
                    modeling.run_model(
                        "action-layouts", layouts / f"{rank}-layouts.json", reference.graphs / f"{rank}-graph.json"
                    )
            except (CalledProcessError, TimeoutError, TimeoutExpired, RuntimeError) as error:
                self.report["layout_error"] = str(error)
                return "action_layout_unavailable"

            inputs = NativeInputs(capture, reference.graphs, timing, layouts, peer_handoffs=args.peer_handoffs)
            extracted = inputs.run(reference)

        stages = self.report
        stages["wall_ns"]["timing"] = monotonic_ns() - started
        stages["timing_status"] = extracted["status"]
        stages["timing_scope"] = extracted["timing_scope"]

        if extracted["status"] != "extracted":
            return "timing_unavailable"
        return inputs
