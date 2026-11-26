"""Qualify worker observations and scheduler batch operands against their native inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import WorkCatalog
from prologue.common.io import read_json, write_json
from prologue.host.libraries.release import asset_path
from prologue.integrations.bindings import device_runtime, platform_domain

from .reference import observed_scopes


class WorkerCapture:
    def __init__(self, recording, rank, part=None, *, cut=None):
        self.recording, self.rank = recording, rank
        self.job, revision = recording.inputs["job"], recording.inputs["framework_revision"]
        self.source, self.manifest = recording.source, recording.manifest

        if cut is not None:
            self.descend("calibration", cut)
            self.job = self.job.copy()
            self.job["batch"] = self.job.pop("calibration")[cut]

        if part is not None:
            self.descend("probes", part)
            if self.manifest["framework_revision"] != revision:
                raise ValueError("paired local probes must use the same native framework")

        if "workers" in self.manifest:
            self.descend("workers", str(rank))
            if (self.manifest["rank"], self.manifest["framework_revision"]) != (rank, revision):
                raise ValueError("local capture manifest belongs to a different worker")
        elif rank != 0:
            raise ValueError("a single-worker recording contains only rank zero")

    def descend(self, group, key):
        self.source = asset_path(self.source, self.manifest[group][key])
        self.manifest = read_json(self.source / "capture.json")

    @classmethod
    def observed(cls, recording):
        return [cls(recording, int(rank)) for rank in recording.manifest.get("workers", {"0": "."})]

    def read(self, part):
        return read_json(asset_path(self.source, self.manifest[part]))

    def boundary(self):
        result = self.read("boundary")
        if "output" in result:

            # Dummy weights may change token values. Compare slots, then keep the type and location for pairing.
            output = BatchReadback.output(result)
            result["output"] = {k: output[k] for k in ("dtype", "device")}
        return result

    @staticmethod
    def template_resources(facts):
        environment = facts["environment"]
        if environment["parallel"]["logical_pp"] != 1 or environment["implementation"]["disaggregation_mode"] != "null":
            return facts["resources"]

        # Synchronous cuts run in the caller's stream. Both device markers record that actual stream, including
        # around collective transport.
        streams = {facts["spans"][p["at"]["span"]]["resource"] for scope in facts["scopes"] for p in scope["ports"]
                   if p["name"] in {"device.enter", "device.complete"}}
        resources = facts["resources"]
        if len(streams) == 1:
            lane, = streams
            if resources[lane]["kind"] == "stream":
                resources = resources.copy()
                resources[lane] = resources[lane] | {"binding": "synchronous_caller"}

        return resources

    @staticmethod
    def observed_facts(source, rank, trace):
        raw = read_json(source / f"rank-{rank}.json")
        raw["scopes"] = observed_scopes(raw, device_runtime().tensor_layouts)
        return device_runtime().observation(source, rank, raw, trace, evidence=True).document()

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

    def measure(self, model, workspace):
        with TemporaryDirectory(dir=workspace) as scratch:
            return self.catalog(model, Path(scratch))


class BatchReadback:
    @classmethod
    def observed_asset(cls, batch, boundary, facts):
        if boundary["phase"] != batch["phase"]:
            raise ValueError("native batch acquisition ran in a different phase")
        if "output" in boundary:
            cls.check_model(boundary, facts)
            return "structure/model.json"

        if boundary["phase"] == "admission":
            if boundary["waiting"] != [row["id"] for row in batch["requests"]]:
                raise ValueError("native waiting queue differs from the supplied request order")
        elif boundary["phase"] not in {"selection", "scheduler"}:
            cls.check_host(batch, boundary, facts)

        return "structure/host.json"

    @staticmethod
    def geometry(members):
        return [{k: member[k] for k in ("id", "phase", "context_tokens", "new_tokens")} for member in members]

    @staticmethod
    def output(bounds):
        native = bounds["output"]
        if native["tokens"] != native["relayed"] or len(native["tokens"]) != len(bounds["members"]):
            raise ValueError("timed model output did not reach the native batch slots")
        return native

    @classmethod
    def check_model(cls, bounds, facts):
        native = cls.output(bounds)
        if native["device"] != platform_domain().device_type:
            raise ValueError("each sampled device token must reach its original native request slot")

        scopes = facts["scopes"]
        root, = (i for i, scope in enumerate(scopes) if scope["role"] == "run_batch" and scope["parent"] == 0)
        family = [s for i, s in enumerate(scopes) if i == root or s["parent"] == root]

        def tensor(role, direction, name):
            ports = [p for s in family if s["role"] == role for p in s["ports"]
                     if p["direction"] == direction and p["name"] == name]
            if len(ports) != 1:
                raise ValueError(f"one native model cut requires one {role} {direction} port for {name}")

            return ports[0]["tensor"]

        logits = tensor("model", "output", "next_token_logits")
        tokens = tensor("run_batch", "output", "next_token_ids")
        if (tensor("sample", "input", "next_token_logits")["value"] != logits["value"]
                or tensor("sample", "output", "next_token_ids")["value"] != tokens["value"]):
            raise ValueError("model, sampler and batch must hand off the observed tensor values")
        if tokens["shape"] != [len(bounds["members"])] or tokens["dtype"] != native["dtype"]:
            raise ValueError("the observed model output port must describe the returned token tensor")

    @staticmethod
    def check_host(batch, bounds, facts):
        expected = []
        for row, member in zip(batch["requests"], bounds["members"], strict=True):
            prompt = sum(n for _, n in row["prompt"])
            count = 1 if batch["phase"] == "prefill" else member["context_tokens"] - prompt + 2
            expected.append({"id": row["id"], "generated_tokens": count, "finished": count == row["output_tokens"]})

        publication = [
            effect | {"tokens": [row["next_token"]]} for effect, row in zip(expected, batch["requests"], strict=True)
        ]
        emitter = facts["environment"]["parallel"]["tp_rank"] == 0

        if bounds["effects"] != expected or bounds["publication"] != (publication if emitter else []):
            raise ValueError("native request state and receiving socket must agree on each produced token")
        scopes = facts["scopes"]
        bindings = [scope["bindings"] for scope in scopes]
        owners = {b.get("segment", {}).get("owner") for b in bindings}
        produced = [b["publication"] for b in bindings if "publication" in b]
        completed = [b["effects"] for b in bindings if "effects" in b and b["segment"]["owner"] == "result"]
        named = [{k: e[k] for k in ("id", "generated_tokens")} for e in expected]
        required = {"selection", "preparation", "result"} | ({"publication"} if emitter else set())

        if not required <= owners or produced != ([named] if emitter else []) or completed != [expected]:
            raise ValueError("observed host segments must account for the native request and publication effects")

        # The supplied token tensor is on the device. Its materialization must account for the actual batch payload.
        reads = [i for i, s in enumerate(scopes) if s["role"] == "host_read"
                 and s["bindings"]["condition"]["input"] == "next_token_ids"]
        copies = [s for s in facts["spans"] if s["scope"] in reads
                  and s.get("copy", {}).get("operand") == "next_token_ids"]

        if (len(reads) != 1 or len(copies) != 1 or copies[0]["copy"]["direction"] != "device_to_host"
                or copies[0]["operand_bytes"] != len(expected) * copies[0]["copy"]["element_bytes"]
                or not any(p["direction"] == "input" and p["name"] == "next_token_ids"
                           for p in scopes[reads[0]]["ports"])):
            raise ValueError("the host read must consume the supplied device output for each batch member")
