"""Read an immutable native task together with the conditions that produced its evidence."""

import json


class PublishedCapture:
    @classmethod
    def from_source(cls, source):
        source = source.resolve(strict=True)
        recording = cls(source.parents[2])
        if recording.source != source:
            raise ValueError(f"native source is not the published attempt: {source}")
        return recording

    def __init__(self, directory, job=None, *, conditions=None):
        self.directory = directory.resolve(strict=True)
        state = json.loads((self.directory / "execution.json").read_text())
        self.inputs = state["inputs"]
        actual = json.loads((self.directory / "input.json").read_text())
        if self.inputs["job"] != actual or (job is not None and job != actual):
            raise ValueError(f"published capture differs from the planned native input: {self.directory}")
        if conditions and any(self.inputs.get(key) != value for key, value in conditions.items()):
            raise ValueError(f"published capture differs from the requested recording conditions: {self.directory}")

        attempts = state["attempts"]
        if not attempts or attempts[-1]["status"] != "exited" or attempts[-1]["exit_code"] != 0:
            raise ValueError(f"published capture has no successful terminal attempt: {self.directory}")

        # The native launcher gives each attempt a unique container name that survives copying or moving its artifacts.
        self.attempt_id = attempts[-1]["container"]
        publication = json.loads((self.directory / "result.json").read_text())
        self.source = (self.directory / publication["source"]).resolve(strict=True)
        if self.source != self.directory / "attempts" / str(attempts[-1]["number"]) / "source":
            raise ValueError("published source differs from the successful native attempt")

        # The worker closes this manifest before the coordinator publishes the task.
        self.manifest = json.loads((self.source / "capture.json").read_text())
        if (self.manifest["framework_revision"] != self.inputs["framework_revision"]
                or any(publication[key] != value for key, value in self.manifest.items())):
            raise ValueError("published observation differs from its native recording")
