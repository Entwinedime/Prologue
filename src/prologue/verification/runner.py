"""Own the qualification and publication common to one verification case."""

from functools import cached_property

from prologue.runtime.io import write_json
from prologue.verification.core.reference import ReferenceRecording


class VerificationRun:
    success = {"compared"}
    recording_conditions = None

    def __init__(self, arguments):
        self.arguments = arguments
        self.output = arguments.output.resolve()
        self.report = {}

    @cached_property
    def recording(self):
        return ReferenceRecording(self.arguments.capture, conditions=self.recording_conditions)

    def run(self):
        self.output.mkdir(parents=True)
        return self.verify()

    def finish(self, status, **details):
        write_json(self.output / "result.json", {"status": status, **self.report, **details}, indent=2, replace=False)
        return status in self.success

    def training_gap(self, **assets):
        evidence = self.report["training"] = self.recording.training(**assets)
        return evidence["status"] if evidence["status"] != "separate" else None

    def input_mismatch(self, target, *, case=None, sessions=None):
        if case is not None:
            difference = case.input_difference(target)
        elif sessions is not None:
            difference = self.recording.session_difference(target, self.recording.manifest["instances"], sessions)
        else:
            difference = self.recording.input_difference(self.recording.inputs["job"], target)
        if difference:
            self.report["input_difference"] = difference
        return difference
