"""Read an immutable native task together with the conditions that produced its evidence."""

from dataclasses import dataclass
from pathlib import Path

from prologue.common.io import read_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.libraries.release import LibraryRelease, asset_path
from prologue.host.runtime.limits import Usage


class PublishedCapture:
    """Read the successful attempt and verify that its immutable inputs own the publication."""

    @classmethod
    def from_source(cls, source):
        source = source.resolve(strict=True)
        recording = cls(source.parents[2])
        if recording.source != source:
            raise ValueError(f"native source is not the published attempt: {source}")
        return recording

    def __init__(self, directory, job=None, *, conditions=None):
        self.directory = directory.resolve(strict=True)
        state = read_json(self.directory / "execution.json")
        self.inputs = state["inputs"]
        actual = read_json(self.directory / "input.json")

        if self.inputs["job"] != actual or (job is not None and job != actual):
            raise ValueError(f"published capture differs from the planned native input: {self.directory}")
        if conditions and any(self.inputs.get(key) != value for key, value in conditions.items()):
            raise ValueError(f"published capture differs from the requested recording conditions: {self.directory}")

        attempts = state["attempts"]
        if not attempts or attempts[-1]["status"] != "exited" or attempts[-1]["exit_code"] != 0:
            raise ValueError(f"published capture has no successful terminal attempt: {self.directory}")

        # The native launcher gives each attempt a unique container name that survives copying or moving its artifacts.
        self.attempt_id = attempts[-1]["container"]
        publication = read_json(self.directory / "result.json")
        self.source = (self.directory / publication["source"]).resolve(strict=True)
        if self.source != self.directory / "attempts" / str(attempts[-1]["number"]) / "source":
            raise ValueError("published source differs from the successful native attempt")

        # The worker closes this manifest before the coordinator publishes the task.
        self.manifest = read_json(self.source / "capture.json")
        if (self.manifest["framework_revision"] != self.inputs["framework_revision"]
                or any(publication[key] != value for key, value in self.manifest.items())):
            raise ValueError("published observation differs from its native recording")

class SessionCapture(PublishedCapture):
    """Bind a qualified session receipt to the native attempt that supplied its assets."""

    def __init__(self, session, receipt=None):
        self.session = session.resolve(strict=True)
        receipt = read_json(self.session / "result.json") if receipt is None else receipt
        super().__init__(asset_path(self.session, receipt.get("recording", "native")))

        if (receipt["status"] not in {"qualified", "incomplete"} or receipt["native_recording"] != self.attempt_id
                or receipt["rule"] != self.inputs["rule"] or receipt["queries"] != self.inputs["queries"]):
            raise ValueError("qualified session does not own the native recording, rule and queries")
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class LocalSession:
    """Locate native attempts and account for failed or superseded candidates as well as the winner."""

    directory: Path

    def native(self, name, primary):
        parent = self.directory if primary else self.directory / "fallback" / name
        return parent / "native"

    def executions(self):
        primary = self.directory / "native/execution.json"
        if primary.is_file():
            yield primary
        yield from sorted(self.directory.glob("fallback/*/native/execution.json"))

    def attempts(self):
        return (attempt for path in self.executions() for attempt in read_json(path)["attempts"])

    def charges(self, *, excluding=None):
        charges, card_ns = [], 0
        for path in self.executions():
            if path.parent == excluding:
                continue
            execution = read_json(path)
            usage = NativeAttempt.charges(execution["attempts"])
            charges.append(usage)
            card_ns += len(execution["inputs"]["physical_devices"]) * usage.native_ns
        return Usage.total(charges), card_ns


class SourceProductions:
    """Read the source jobs and their published captures and libraries."""

    def __init__(self, collection):
        collection = collection.resolve(strict=True)
        self.directory = collection
        inputs = read_json(collection / "input.json")
        self.plan, self.jobs, self.deployment = inputs["plan"], inputs["jobs"], inputs["deployment"]
        publication = read_json(collection / "result.json")
        self.unavailable = publication.get("unavailable", {})
        planned = {str(job["id"]) for job in self.plan["sources"]}
        jobs = [job for job in self.plan["sources"] if str(job["id"]) not in self.unavailable]

        if (not len(jobs) == len(publication["productions"]) == len(publication["captures"])
                or self.jobs.keys() != planned or self.unavailable.keys() - planned):
            raise ValueError("source publication does not match the planned source jobs")
        self.paths = {job["id"]: collection / name for job, name in zip(jobs, publication["productions"], strict=True)}

        # Explicit observations keep their original owners, which may live outside this collection.
        self.captures = {job["id"]: collection / name for job, name in zip(jobs, publication["captures"], strict=True)}

    def publish(self, output, costs):
        observations, identities = {}, {}
        for job, directory in self.paths.items():
            directory = directory.resolve(strict=True)
            recording = PublishedCapture(self.captures[job], self.jobs[str(job)])
            inputs = read_json(directory / "production.json")["inputs"]

            if Path(inputs["capture_path"]).resolve(strict=True) != recording.source:
                raise ValueError(f"template production belongs to another native recording: {directory}")

            result = read_json(directory / "result.json")
            paths = []
            for rank, name in zip(result["observed_roles"], result["templates"], strict=True):
                sampling = result.get("sampling", {}).get(str(rank))
                paths.append((asset_path(directory, name), asset_path(directory, sampling) if sampling else None))

            identity = identities[job] = recording.attempt_id
            value = recording.inputs["deployment"], paths
            if observations.setdefault(identity, value) != value:
                raise ValueError(f"conflicting template productions for recording {identity}")

        # Translate the archived collection's role declarations once; requests and capture budgets stay with the task.
        bindings = []
        jobs = [job for job in self.plan["sources"] if job["id"] in identities]
        for job in jobs:
            if not job["configurations"]:
                continue
            imports = [identities[local["id"]] for local in jobs
                       if any(row["source"] == job["id"] for row in local.get("local_consumers", []))]
            bindings.append({"domain": job["structure"], "workers": identities[job["id"]], "imports": imports})

        return LibraryRelease.publish(output, observations, bindings, costs)
