"""Verify a serving structure against one observed native case."""

import argparse
import json
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from tempfile import TemporaryDirectory
from time import monotonic_ns
from types import SimpleNamespace

from prologue.app.prediction import PredictionProcess
from prologue.artifacts import write_json
from prologue.integrations.runtimes.docker import ModelingProcess

from .compare_actions import ActionComparison
from .execution_inputs import NativeInputs
from .reference import ReferenceRecording
from .source_evidence import ProductionEvidence


class StructureVerification:
    def __init__(self, arguments):
        self.arguments = arguments
        self.output = arguments.output.resolve()

    def finish(self, status, stages):
        write_json(self.output / "result.json", {"status": status, **stages}, indent=2, replace=False)
        return status in {"full_structure_matched", "structure_matched_within_scope"}

    def run(self):
        args = self.arguments
        self.output.mkdir(parents=True)
        recording = ReferenceRecording(args.capture)
        capture = recording.source
        wall_ns = {}
        stages = {
            "capture": str(recording.directory), "reference": str(args.reference.resolve(strict=True)),
            "production": str(args.production.resolve()), "wall_ns": wall_ns,
        }
        source = json.loads(args.input.read_text())
        if difference := recording.input_difference(recording.inputs["job"], source):
            return self.finish("input_mismatch", stages | {"input_difference": difference})
        sources = PredictionProcess.sources(args.production)
        if args.peer_handoffs and sources.deployment["acquisition_mode"] != "single_device":
            return self.finish("peer_handoff_source_unqualified", stages)
        training = stages["training"] = recording.training(production=sources, target=source)
        if training["status"] != "separate":
            return self.finish(training["status"], stages)

        started = monotonic_ns()
        checker = ProductionEvidence(args.reference, capture, self.output / "source-evidence", args.details)
        reference, evidence = checker.inspect()
        stages["reference_status"] = evidence["native_case"]["status"]
        status = checker.publish(evidence, require_joint=True)
        wall_ns["reference"] = monotonic_ns() - started
        if status:
            return self.finish("reference_unqualified", stages)

        if difference := reference.input_difference(source):
            return self.finish("input_mismatch", stages | {"input_difference": difference})

        started = monotonic_ns()
        timing = self.output / "timing"
        with TemporaryDirectory(prefix=".action-layouts-", dir=self.output) as temporary:
            layouts = Path(temporary)
            modeling = ModelingProcess(args.binary, args.modeling_image, args.timeout)
            try:
                for rank in reference.sources:
                    modeling.run_model("action-layouts", layouts / f"{rank}-layouts.json", reference.graphs / f"{rank}-graph.json")
            except (CalledProcessError, TimeoutError, TimeoutExpired, RuntimeError) as error:
                return self.finish("action_layout_unavailable", stages | {"layout_error": str(error)})
            inputs = NativeInputs(capture, reference.graphs, timing, layouts, peer_handoffs=args.peer_handoffs)
            extracted = inputs.run(reference)
        wall_ns["timing"] = monotonic_ns() - started
        stages["timing_status"] = extracted["status"]
        stages["timing_scope"] = extracted["timing_scope"]
        if extracted["status"] != "extracted":
            return self.finish("timing_unavailable", stages)

        prediction = self.output / "prediction"
        command = SimpleNamespace(input=args.input, production=args.production, output=prediction, timeout=args.timeout,
                                  binary=args.binary, modeling_image=args.modeling_image, control_image=args.control_image,
                                  execution_inputs=timing / "execution-inputs.json", costs=None, execution_trace=True)
        started = monotonic_ns()
        try:
            PredictionProcess(command).run(source=source, productions=sources)
        except (RuntimeError, TimeoutError, TimeoutExpired) as error:
            stages["prediction_error"] = str(error)
        wall_ns["prediction"] = monotonic_ns() - started
        trace = prediction / ("execution.partial.jsonl" if "prediction_error" in stages else "execution.jsonl")
        if not trace.is_file():
            return self.finish("prediction_no_trace", stages)

        # Compare against the original reference, including actual condition consumption and independent causal evidence.
        started = monotonic_ns()
        try:
            report = ActionComparison(reference, trace, self.output / "comparison", args.details,
                                      realization=inputs.realization, peer_handoffs=args.peer_handoffs).run()
        except ValueError as error:
            wall_ns["comparison"] = monotonic_ns() - started
            return self.finish("comparison_unverified", stages | {"comparison_error": str(error)})
        wall_ns["comparison"] = monotonic_ns() - started
        stages["comparison_status"] = report["status"]
        stages["coverage"] = report["coverage"]
        stages["actions_compared"] = report["actions_compared"]
        if "prediction_error" in stages:
            if rejected := report["first_rejected_action"]:
                stages["first_rejected_action"] = rejected
                return self.finish("measured_layout_difference", stages)
            if gap := PredictionProcess.gap(prediction):
                status, path = gap
                return self.finish(status, stages | {"evidence": str(path)})
            return self.finish("prediction_failed", stages)
        if report["status"] != "actions_matched":
            return self.finish(report["status"], stages)
        if report["condition_inputs"]["full_node_coverage"]:
            return self.finish("full_structure_matched", stages)
        return self.finish("structure_matched_within_scope", stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="published detailed native capture task")
    for name in ("reference", "production", "input", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True, help="allowance in seconds for each of layout preparation and prediction")
    parser.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    parser.add_argument("--details", action="store_true", help="retain all evidence findings")
    parser.add_argument("--peer-handoffs", action="store_true", help="allow measured common ports when a single-device action has a different layout")
    raise SystemExit(0 if StructureVerification(parser.parse_args()).run() else 1)


if __name__ == "__main__":
    main()
