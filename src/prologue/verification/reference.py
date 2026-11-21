"""Qualify the target input and its separation from the recordings used for training."""

import json
from pathlib import Path

from prologue.acquisition.recording import PublishedCapture
from prologue.integrations.frameworks.sglang.configuration import CaptureInput, ServingInput


class ReferenceRecording(PublishedCapture):
    target_fields = ("model", "instances", "workload")

    @classmethod
    def input_difference(cls, job, target):
        if set(target) != set(cls.target_fields):
            return "target fields"
        for field in cls.target_fields:
            if job[field] != target[field]:
                return field

        native = CaptureInput(job)
        order = [row["id"] for row in native.workload["requests"]]
        return None if native.submission_groups == [order] else "native submission groups"

    @staticmethod
    def sessions(capture, instances):
        return [json.loads((capture / "instances" / str(index) / "session.json").read_text())
                for index, _ in enumerate(instances)]

    @staticmethod
    def session_difference(source, instances, sessions):
        target = ServingInput(source)
        if len(instances) != len(target.instances):
            return "instances"

        for observed, session, instance in zip(instances, sessions, target.instances, strict=True):
            captured = dict(session["serving_input"])
            configuration = captured.pop("configuration")
            if ((observed["id"], observed["mode"]) != (instance.identity, instance.mode)
                    or configuration != instance.configuration.values):
                return f"instance:{instance.identity}"

            # Native content and routing hooks consume separate fields from this recorded session.
            if session["content"] != target.workload or captured != target.workload:
                return "workload"
        return None

    def training(self, *, production=None, target=None, costs=None):
        shared, unknown = [], []
        templates, samples = 0, 0
        if production is not None:
            jobs = production.jobs_for(target)
            if not jobs:
                return {"target_attempt": self.attempt_id, "template_jobs": 0, **production.failure(target)}

            templates = len(jobs)
            for job in jobs:
                path = production.paths[job] / "production.json"
                try:
                    source = Path(json.loads(path.read_text())["inputs"]["capture_path"])
                    recording = PublishedCapture.from_source(source)
                except (OSError, KeyError, ValueError) as error:
                    unknown.append({"template_job": job, "reason": str(error)})
                    continue

                if recording.attempt_id == self.attempt_id:
                    shared.append({"template_job": job})
                if recording.directory != production.captures[job].resolve() or recording.inputs["job"] != production.jobs[str(job)]:
                    unknown.append({"template_job": job, "reason": "production and collection name different recordings"})

        if costs is not None:
            asset = json.loads(costs.read_text())
            sources = asset.get("sources", [])
            origins = asset.get("sample_recordings", [])
            samples = len(sources)
            if not samples or not isinstance(origins, list) or len(origins) != samples:
                unknown.append({"cost_asset": str(costs), "reason": "sample recordings unavailable"})
            else:
                for index, recordings in enumerate(origins):
                    if not isinstance(recordings, list) or not recordings or not all(isinstance(r, str) and r for r in recordings):
                        unknown.append({"sample": index, "source": sources[index]})
                    elif self.attempt_id in recordings:
                        shared.append({"sample": index, "source": sources[index]})

        status = "reference_used_for_training" if shared else "training_origin_unverified" if unknown else "separate"
        return {"status": status, "target_attempt": self.attempt_id, "template_jobs": templates, "sample_files": samples,
                "shared": shared, "unknown": unknown}
