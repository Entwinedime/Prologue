"""Locate one observed logical worker inside a qualified local scheduler recording."""

import json

from prologue.runtime.io import write_json
from prologue.integrations.frameworks.sglang.observation.source_scopes import SGLangSourceScopes
from prologue.integrations.platforms.ascend.observation import AscendObservation
from prologue.libraries.release import asset_path
from prologue.performance.samples import WorkCatalog


class WorkerCapture:
    def __init__(self, recording, rank, part=None, *, cut=None):
        self.recording, self.rank = recording, rank
        self.job = recording.inputs["job"]
        source, manifest = recording.source, recording.manifest
        if cut is not None:
            source = asset_path(source, manifest["calibration"][cut])
            manifest = json.loads((source / "capture.json").read_bytes())
            self.job = {k: v for k, v in self.job.items() if k != "calibration"} | {"batch": self.job["calibration"][cut]}
        if part is not None:
            source = asset_path(source, manifest["probes"][part])
            manifest = json.loads((source / "capture.json").read_bytes())
            if manifest["framework_revision"] != recording.inputs["framework_revision"]:
                raise ValueError("paired local probes must use the same native framework")
        workers = manifest.get("workers")
        if workers is None:
            if rank != 0:
                raise ValueError("a single-worker recording contains only rank zero")
            self.source, self.manifest = source, manifest
        else:
            self.source = asset_path(source, workers[str(rank)])
            self.manifest = json.loads((self.source / "capture.json").read_bytes())
            if self.manifest["rank"] != rank or self.manifest["framework_revision"] != recording.inputs["framework_revision"]:
                raise ValueError("local capture manifest belongs to a different worker")

    @classmethod
    def observed(cls, recording):
        return [cls(recording, int(rank)) for rank in recording.manifest.get("workers", {"0": "."})]

    def read(self, part):
        return json.loads(asset_path(self.source, self.manifest[part]).read_bytes())

    def boundary(self):
        result = self.read("boundary")
        if "output" in result:

            # Dummy weights can change token values. Both native cuts must still fill the same request slots.
            output = result["output"]
            if output["tokens"] != output["relayed"] or len(output["tokens"]) != len(result["members"]):
                raise ValueError("timed model output did not reach the native batch slots")
            result["output"] = {k: output[k] for k in ("dtype", "device")}
        return result

    @staticmethod
    def template_resources(facts):
        environment = facts["environment"]
        if environment["parallel"]["logical_pp"] != 1 or environment["implementation"]["disaggregation_mode"] != "null":
            return facts["resources"]

        # Synchronous cuts run in the caller's stream. Both device markers record that actual stream, including around HCCL.
        streams = {facts["spans"][port["at"]["span"]]["resource"] for scope in facts["scopes"] for port in scope["ports"]
                   if port["name"] in {"device.enter", "device.complete"}}
        resources = facts["resources"]
        if len(streams) != 1:
            return resources
        lane, = streams
        if resources[lane]["kind"] != "stream":
            return resources
        return [resource | {"binding": "synchronous_caller"} if i == lane else resource for i, resource in enumerate(resources)]

    @staticmethod
    def observed_facts(source, rank, trace):
        raw = json.loads((source / f"rank-{rank}.json").read_bytes())
        raw["scopes"] = SGLangSourceScopes(raw["environment"]).lower(raw["scopes"])
        return AscendObservation(source, rank, raw, trace, evidence=True).document()

    @staticmethod
    def project_work(facts, model, directory):
        native, graph, work = (directory / name for name in ("facts.json", "graph.json", "work.json"))
        write_json(native, facts)
        model.run_model("project", graph, native, options=("--evidence",))
        model.run_model("work-identities", work, graph)
        return WorkCatalog(graph, work)

    def catalog(self, model, directory):
        facts = self.observed_facts(self.source, self.rank, self.source / "sdk/trace.json")
        return facts, self.project_work(facts, model, directory)
