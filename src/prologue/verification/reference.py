"""Qualify the target input and its separation from the recordings used for training."""

import json
from pathlib import Path

from prologue.acquisition.recording import PublishedCapture
from prologue.integrations.frameworks.sglang.configuration import CaptureInput, ServingInput


class RecordedSubmissions:
    clock = "monotonic_raw_ns"

    @classmethod
    def read(cls, capture, sessions):
        records = []
        for index, session in enumerate(sessions):
            path = capture / "instances" / str(index) / "request-delivery.json"
            delivery, = json.loads(path.read_text())["capture"]
            records.append(cls(session, delivery))
        return records

    def __init__(self, session, delivery):
        if delivery["clock"] != self.clock:
            raise ValueError("native request submissions require the shared RAW clock")
        self.origin_ns = delivery["origin_ns"]
        requests = session["content"]["requests"]
        self.requests = {request["id"]: request for request in requests}
        self.aliases = dict(zip(session["source_requests"], self.requests, strict=True))
        if len(self.aliases) != len(requests):
            raise ValueError("each submitted request needs a unique native and workload identity")

        # The driver records a single submission time for all requests in one message.
        pairs = [(index, row) for row in delivery["submissions"] for index in row["requests"]]
        if sorted(index for index, _ in pairs) != list(range(len(requests))):
            raise ValueError("native submissions must cover every request exactly once")
        self.by_request = {requests[index]["id"]: row for index, row in pairs}

    def receives(self, records):
        rounds, received = [], set()
        for record in records:
            time_ns = record["received_raw_ns"] - self.origin_ns
            members = [self.aliases[name] for name in record["requests"]]
            if not members or time_ns < 0 or (rounds and time_ns < rounds[-1]["time_ns"]):
                raise ValueError("receive rounds require nonempty members and nondecreasing times")
            for member in members:
                if member in received or time_ns < max(self.requests[member]["arrival_ns"], self.by_request[member]["submitted_ns"]):
                    raise ValueError(f"receive repeats {member} or precedes its submission")
                received.add(member)
            rounds.append({"time_ns": time_ns, "requests": members})
        if received != self.requests.keys():
            raise ValueError("receive rounds do not cover every request exactly once")
        return rounds


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
