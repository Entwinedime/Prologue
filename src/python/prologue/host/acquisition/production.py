"""Project a closed native capture, optionally compiling representative templates."""

import subprocess
from uuid import uuid4

from prologue.common.io import discard_path, read_json, write_json
from prologue.common.observation.workers import CapturedWorkers
from prologue.host.acquisition.adapters import ProductionAdapter
from prologue.host.acquisition.options import ProductionOptions
from prologue.host.runtime.docker import DockerImage, OwnedContainers
from prologue.host.runtime.limits import AllowanceExhausted, OperationLimit, Usage
from prologue.host.runtime.workspace import BinarySnapshot, TaskWorkspace, file_lock


class CaptureProduction:
    @staticmethod
    def usage(directory):
        state = directory / "production.json"
        return Usage(processing_ns=read_json(state)["spent_ns"]) if state.is_file() else Usage()

    def __init__(self, arguments: ProductionOptions, recording, adapter: type[ProductionAdapter], limit=None) -> None:
        self.arguments, self.adapter = arguments, adapter
        self.operation_limit = limit if limit is not None else OperationLimit()
        self.output = arguments.output.resolve()
        self.capture = recording.source
        self.binary = arguments.binary.resolve(strict=True)
        if self.output.is_relative_to(recording.directory):
            raise ValueError("production must not write inside its immutable native task")
        self.limit_ns = int(arguments.budget_seconds * 1_000_000_000)
        if self.limit_ns <= 0:
            raise ValueError("postprocessing budget must be positive")

        if recording.inputs.get("latency_only"):
            raise ValueError("source production requires a detailed native recording")
        capture = recording.manifest
        self.ranks = CapturedWorkers(capture["instances"]).observed
        self.sampling_input = (
            adapter.sampling_input(recording.inputs["job"]) if arguments.product == "templates" else None
        )
        if not self.ranks:
            raise ValueError("source production requires at least one profiled worker")
        images = {}
        for role, image in (("source", arguments.source_image), ("modeling", arguments.modeling_image)):
            images[role] = DockerImage.inspect(image)
        if adapter.image_revision(images["source"]) != capture["framework_revision"]:
            raise ValueError("source postprocessing image differs from the captured framework revision")
        self.inputs = {"capture_path": str(self.capture), "capture": capture, "native_image": recording.inputs["image"],
                       "images": {role: image["Id"] for role, image in images.items()},
                       "product": arguments.product, "keep_intermediates": arguments.keep_intermediates}

    def publish_state(self) -> None:
        write_json(self.output / "production.json", self.state, indent=2)

    @classmethod
    def release_work(cls, output, graph_ranks=()):
        work = output / "work"
        if not graph_ranks:
            discard_path(work)
            return

        published = {f"{rank}-graph.json" for rank in graph_ranks}
        for path in work.iterdir():
            if path.name not in published:
                discard_path(path)

    @classmethod
    def retire_projection(cls, output):

        # The cost task has consumed its private projection. Its original processing charge stays auditable.
        if output.is_dir():
            for path in output.iterdir():
                if path.name not in {"production.json", ".lock"}:
                    discard_path(path)

    def prepare(self) -> None:
        snapshot = BinarySnapshot(self.binary)
        with TaskWorkspace(self.output).initialize("production.json") as draft:
            if draft is not None:
                for name in ("inputs", "work", "logs"):
                    (draft / name).mkdir()
                snapshot.copy_to(draft / "inputs/prologue")
                if self.arguments.product == "templates":
                    templates = [f"/production/work/{rank}-templates.json" for rank in self.ranks]
                    write_json(draft / "inputs/templates.json", templates)
                if self.sampling_input:
                    write_json(draft / "inputs/sampling.json", self.sampling_input)
                self.state = {"inputs": self.inputs, "spent_ns": 0, "attempts": 0,
                              "completed": [], "running": None, "error": None}
                write_json(draft / "production.json", self.state, indent=2)
                return
        self.state = read_json(self.output / "production.json")
        if self.state["inputs"] != self.inputs:
            raise ValueError("production inputs changed; use a new output directory")
        snapshot.check(self.output / "inputs/prologue")
        if running := self.state["running"]:

            # Only the unfinished step's container and output belong to recovery.
            OwnedContainers([running["container"]]).close()
            discard_path(self.output / running["output"])
            self.state.update(running=None, error="interrupted operation charged its reserved remaining budget")
            self.publish_state()

    def step(self, name: str, output: str, image: str, command: list[str]) -> None:
        if name in self.state["completed"]:
            if not (self.output / output).exists():
                raise FileNotFoundError(f"completed production step lost its output: {output}")
            return
        remaining_ns = self.limit_ns - self.state["spent_ns"]
        if remaining_ns <= 0:
            self.state["error"] = "cumulative postprocessing budget exhausted; completed steps remain reusable"
            self.publish_state()
            raise AllowanceExhausted(self.state["error"])
        self.operation_limit.check()
        container = f"prologue-produce-{uuid4().hex}"
        self.state["attempts"] += 1
        log = self.output / "logs" / f"{self.state['attempts']:03d}-{name}.log"

        # Reserve before starting: hard interruption cannot reset the cumulative allowance on resume.
        self.state["running"] = {"container": container, "step": name, "output": output}
        self.state["error"] = None
        with TaskWorkspace.reserve(self.state, "spent_ns", remaining_ns, self.publish_state,
                                   settled=lambda: self.state["running"] is None):
            try:
                with log.open("x") as stream:

                    # The cumulative reservation also bounds the container after coordinator interruption.
                    command = [
                        "docker",
                        "run",
                        "--rm",
                        "--init",
                        "--runtime",
                        "runc" if image == "modeling" else self.adapter.platform_runtime().source_docker_runtime(),
                        "--network",
                        "none",
                        "--name",
                        container,
                        "--entrypoint",
                        "/usr/bin/timeout",
                        "-e",
                        "LD_PRELOAD=",
                        "--mount",
                        f"type=bind,src={self.capture},dst=/source,readonly",
                        "--mount",
                        f"type=bind,src={self.output},dst=/production",
                        self.inputs["images"][image],
                        "--signal=KILL",
                        f"{remaining_ns / 1_000_000_000}s",
                        *command,
                    ]
                    code = self.operation_limit.run(command, stream, remaining_ns / 1_000_000_000)
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                if not (self.output / output).exists():
                    raise FileNotFoundError(f"production command did not publish {output}")
                self.state["completed"].append(name)
            except BaseException as error:
                self.state["error"] = f"{name}: {type(error).__name__}: {error}"
                raise
            finally:
                OwnedContainers([container]).close()
                if name not in self.state["completed"]:
                    discard_path(self.output / output)
                self.state["running"] = None

    def run(self) -> None:
        with file_lock(self.output / ".lock"):
            self.prepare()
            if (self.output / "result.json").is_file():
                return

            templates = self.arguments.product == "templates"
            evidence = self.arguments.keep_intermediates or self.sampling_input is not None
            pending = [] if "link-templates" in self.state["completed"] else self.ranks
            for rank in pending:
                facts, graph, template = (
                    f"work/{rank}-facts.json",
                    f"work/{rank}-graph.json",
                    f"work/{rank}-templates.json",
                )
                exported = f"work/{rank}-sdk"
                terminal = f"{rank}-templates" if templates else f"{rank}-project"
                if terminal not in self.state["completed"]:
                    self.step(
                        f"{rank}-export",
                        exported,
                        "source",
                        self.adapter.export_profile(rank, f"/production/{exported}"),
                    )
                    command = self.adapter.interpret(rank, f"/production/{exported}/trace.json", f"/production/{facts}",
                                                     evidence=evidence)
                    self.step(f"{rank}-interpret", facts, "source", command)
                self.step(f"{rank}-project", graph, "modeling", [
                    "/production/inputs/prologue", "project", f"/production/{facts}", f"/production/{graph}",
                    *(["--evidence"] if evidence else []),
                ])
                if templates:
                    self.step(f"{rank}-templates", template, "modeling", [
                        "/production/inputs/prologue", "templates", f"/production/{graph}", f"/production/{template}",
                    ])
                if self.sampling_input:
                    identities = f"work/{rank}-identities.json"
                    self.step(
                        f"{rank}-work",
                        identities,
                        "modeling",
                        [
                            "/production/inputs/prologue",
                            "work-identities",
                            f"/production/{graph}",
                            f"/production/{identities}",
                        ],
                    )
                    cuts = f"sampling/{rank}.json"
                    command = self.adapter.sampling_cuts(rank, f"/production/{identities}", f"/production/{cuts}")
                    self.step(f"{rank}-cuts", cuts, "source", command)

            # Observed target graphs remain valid independently of source-template selection and transformation rules.
            if templates:
                self.step(
                    "link-templates",
                    "assets",
                    "modeling",
                    [
                        "/production/inputs/prologue",
                        "link-templates",
                        "/production/inputs/templates.json",
                        "/production/assets",
                    ],
                )
            if not self.arguments.keep_intermediates:
                self.release_work(self.output, () if templates else self.ranks)

            # Publish only when every observed worker has its requested product; graphs cannot be consumed as templates.
            write_json(
                self.output / "result.json",
                {
                    self.arguments.product: [
                        f"assets/{rank}.json" if templates else f"work/{rank}-graph.json" for rank in self.ranks
                    ],
                    **(
                        {"sampling": {str(rank): f"sampling/{rank}.json" for rank in self.ranks}}
                        if self.sampling_input
                        else {}
                    ),
                    "observed_roles": self.ranks,
                    "postprocessing_ns": self.state["spent_ns"],
                    "qualification": "compiled_source_templates" if templates else "observed_coarse_graphs",
                },
                indent=2,
            )
        print(f"Created {self.output / 'result.json'}")
