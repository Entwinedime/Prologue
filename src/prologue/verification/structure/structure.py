"""Verify a serving structure against one observed native case."""

import argparse
import json
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from tempfile import TemporaryDirectory
from time import monotonic_ns

from prologue.workflow.prediction import PredictionProcess
from prologue.libraries.release import LibraryRelease
from prologue.runtime.docker import ModelingProcess
from prologue.workflow.options import PredictionOptions

from .compare_actions import ActionComparison
from .execution_inputs import NativeInputs
from prologue.verification.runner import VerificationRun
from ..core.source_evidence import ProductionEvidence


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
            "capture": str(recording.directory), "reference": str(args.reference.resolve(strict=True)),
            "libraries": str(args.libraries.resolve()), "wall_ns": wall_ns,
        }
        source = json.loads(args.input.read_text())
        if self.input_mismatch(source):
            return self.finish("input_mismatch")
        prediction = self.output / "prediction"
        timing = self.output / "timing"
        options = PredictionOptions(output=prediction, timeout=args.timeout, binary=args.binary,
                                  modeling_image=args.modeling_image, control_image=args.control_image,
                                  execution_inputs=timing / "execution-inputs.json", execution_trace=True)
        process = PredictionProcess(options, LibraryRelease(args.libraries), source, self.adapter)
        binding = process.task.binding
        if binding is None:
            return self.finish("template_gap", gap=process.task.missing_workers())
        if args.peer_handoffs and binding.worker_deployment.get("acquisition_mode", "full") != "single_device":
            return self.finish("peer_handoff_source_unqualified")
        if status := self.training_gap(templates=binding):
            return self.finish(status)

        started = monotonic_ns()
        checker = ProductionEvidence(args.reference, capture, self.output / "source-evidence", args.details)
        reference, evidence = checker.inspect()
        stages["reference_status"] = evidence["native_case"]["status"]
        status = checker.publish(evidence, require_joint=True)
        wall_ns["reference"] = monotonic_ns() - started
        if status:
            return self.finish("reference_unqualified")

        if self.input_mismatch(source, case=reference):
            return self.finish("input_mismatch")

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
                return self.finish("action_layout_unavailable", layout_error=str(error))
            inputs = NativeInputs(capture, reference.graphs, timing, layouts, peer_handoffs=args.peer_handoffs)
            extracted = inputs.run(reference)
        wall_ns["timing"] = monotonic_ns() - started
        stages["timing_status"] = extracted["status"]
        stages["timing_scope"] = extracted["timing_scope"]
        if extracted["status"] != "extracted":
            return self.finish("timing_unavailable")

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
            report = ActionComparison(reference, trace, self.output / "comparison", args.details,
                                      realization=inputs.realization, peer_handoffs=args.peer_handoffs).run()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="published detailed native capture task")
    for name in ("reference", "libraries", "input", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="allowance in seconds for each of layout preparation and prediction",
    )
    parser.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    parser.add_argument("--details", action="store_true", help="retain all evidence findings")
    parser.add_argument(
        "--peer-handoffs",
        action="store_true",
        help="allow measured common ports when a single-device action has a different layout",
    )
    from ..core.adapters import SGLangTask
    raise SystemExit(0 if StructureVerification(parser.parse_args(), SGLangTask).run() else 1)


if __name__ == "__main__":
    main()
