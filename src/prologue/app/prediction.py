"""Own one prediction process and its isolated core/control lifecycle."""

import json
import math
import os
import subprocess
import time
from contextlib import ExitStack
from uuid import uuid4

from prologue.integrations.frameworks.sglang.acquisition.worker_templates import SGLangTask, WorkerDomain
from prologue.integrations.frameworks.sglang.shared.configuration import ServingInput
from prologue.integrations.frameworks.sglang.shared.ingress import BatchOperands
from prologue.runtime.docker import DockerImage, OwnedContainers
from prologue.runtime.io import content_id, write_json
from prologue.runtime.limits import OperationLimit
from prologue.workflow.options import PredictionOptions


class PredictionProcess:
    def __init__(self, arguments: PredictionOptions, release, source, limit=None) -> None:
        self.arguments = arguments
        self.release = release
        self.limit = limit if limit is not None else OperationLimit()

        self.given_work_ns = source.get("given_work_ns")
        if "given_work_ns" in source and (
            type(self.given_work_ns) is not int or not 0 <= self.given_work_ns < 1 << 63 or arguments.execution_inputs
        ):
            raise ValueError(
                "given_work_ns requires nonnegative signed 64-bit nanoseconds and no measured timing input"
            )
        self.source = {key: value for key, value in source.items() if key != "given_work_ns"}
        self.task = SGLangTask(self.source, release.templates)

    @staticmethod
    def recover(output):
        path = output / "containers.json"
        if path.exists():
            OwnedContainers(json.loads(path.read_text())).close()

    @staticmethod
    def basis(source, release):
        task = SGLangTask(source, release.templates)
        if task.binding is None:
            return content_id(["missing_workers", task.domain])

        workers = {content_id(WorkerDomain.execution_domain(person["environment"]))
                   for instance in task.serving.instances for person in task.participants(instance)}
        costs = {worker: release.cost_identities.get(worker) for worker in workers}
        return content_id([release.templates.program_identity(task.binding), costs])

    @staticmethod
    def gap(output, source):
        for status, name in (("template_gap", "template-gap.json"), ("cost_gap", "cost-gap.json")):
            evidence = output / name
            if evidence.is_file():
                detail = json.loads(evidence.read_bytes())
                action = detail.get("invocation", detail)
                if "native_inputs" not in action and BatchOperands.bind(action, ServingInput(source), detail["domain"]):
                    write_json(evidence, detail)
                return status, evidence
        return None

    def run(self) -> bool:
        args = self.arguments
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("Prediction timeout must be positive")
        self.limit.check()
        deadline = time.monotonic() + args.timeout
        output = args.output.resolve()
        if output.is_relative_to(self.release.root):
            raise ValueError("prediction output must be outside the published libraries")
        output.mkdir(parents=True)
        if self.task.binding is None:
            write_json(output / "template-gap.json", self.task.missing_workers(), replace=False)
            return False

        document = self.task.document()
        timing_mount, timing_option = [], []
        if self.given_work_ns is not None:
            document["given_work_ns"] = self.given_work_ns
        else:
            timing_asset = args.execution_inputs or self.release.costs
            option = "--execution-inputs" if args.execution_inputs else "--costs"
            path = f"/inputs/{option[2:]}.json"
            timing_mount = ["--mount", f"type=bind,src={timing_asset.resolve(strict=True)},dst={path},readonly"]
            timing_option = [option, path]
        binary = args.binary.resolve(strict=True)
        descriptions = {
            role: DockerImage.inspect(image, timeout=max(0, deadline - time.monotonic()))
            for role, image in (("modeling", args.modeling_image), ("control", args.control_image))
        }
        WorkerDomain.require_control(descriptions["control"], self.task.workers)
        images = {role: image["Id"] for role, image in descriptions.items()}
        definitions = self.task.binding.definitions
        definition_mount = (
            ["--mount", f"type=bind,src={definitions},dst=/inputs/templates/definitions,readonly"]
            if definitions
            else []
        )
        task = output / ".task.json"
        stem = f"prologue-serve-{uuid4().hex}"
        names = [f"{stem}-core", f"{stem}-control"]
        write_json(output / "containers.json", names, replace=False)
        try:
            with task.open("x") as stream:
                json.dump(document, stream)
                stream.write("\n")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Prediction preparation exhausted its time allowance")

            # The core alone receives the selected timing asset.
            commands = [
                [
                    "docker",
                    "run",
                    "--rm",
                    "--init",
                    "-i",
                    "--runtime",
                    "runc",
                    "--network",
                    "none",
                    "--name",
                    names[0],
                    "--entrypoint",
                    "/usr/bin/timeout",
                    "--mount",
                    f"type=bind,src={binary},dst=/app/prologue,readonly",
                    *[
                        item
                        for index, path in enumerate(self.task.binding.paths)
                        for item in ("--mount", f"type=bind,src={path},dst=/inputs/templates/{index}.json,readonly")
                    ],
                    *definition_mount,
                    "--mount",
                    f"type=bind,src={task},dst=/inputs/task.json,readonly",
                    *timing_mount,
                    "--mount",
                    f"type=bind,src={output},dst=/output",
                    images["modeling"],
                    "--signal=KILL",
                    f"{remaining}s",
                    "/app/prologue",
                    "serve",
                    "/inputs/templates",
                    "/inputs/task.json",
                    "/output/.result.json",
                    *(["--trace", "/output/execution.partial.jsonl"] if args.execution_trace else []),
                    *timing_option,
                ],
                [
                    "docker",
                    "run",
                    "--rm",
                    "--init",
                    "-i",
                    "--runtime",
                    "runc",
                    "--network",
                    "none",
                    "--shm-size",
                    "512m",
                    "--name",
                    names[1],
                    "--entrypoint",
                    "/usr/bin/timeout",
                    images["control"],
                    "--signal=KILL",
                    f"{remaining}s",
                    "python3",
                    "-m",
                    "prologue.app.control_worker",
                    "--kernel-python",
                    "/opt/prologue/sglang/sgl-kernel/python",
                ],
            ]
            try:
                self.execute(commands, names, output, deadline)
            except RuntimeError:
                if self.gap(output, self.source):
                    return False
                raise
            self.limit.check()

            # Both endpoints and their owned containers must close before the result becomes consumable.
            if args.execution_trace:
                os.link(output / "execution.partial.jsonl", output / "execution.jsonl")
                (output / "execution.partial.jsonl").unlink()
            os.link(output / ".result.json", output / "result.json")
        finally:
            (output / ".result.json").unlink(missing_ok=True)
            task.unlink(missing_ok=True)
        print(f"Created {output / 'result.json'}")
        return True

    def execute(self, commands, names, output, deadline) -> None:
        processes = []
        try:
            with ExitStack() as logs, ExitStack() as pipes:
                core_read, control_write = os.pipe()
                control_read, core_write = os.pipe()
                for descriptor in (core_read, control_write, control_read, core_write):
                    pipes.callback(os.close, descriptor)

                # Direct pipes carry the protocol; the application never parses or rewrites native decisions.
                for command, reader, writer, name in zip(
                    commands, (core_read, control_read), (core_write, control_write), ("core", "control"), strict=True
                ):
                    log = logs.enter_context((output / f"{name}.log").open("x"))
                    log.write(json.dumps(command) + "\n")
                    log.flush()
                    processes.append(subprocess.Popen(command, stdin=reader, stdout=writer, stderr=log))
                pipes.close()
                for process, name in zip(processes, names, strict=True):
                    code = self.limit.wait(process, deadline)
                    if code:
                        raise RuntimeError(f"{name} exited with {code}; inspect {output}")

        finally:

            # Docker clients can exit before their containers; remove only this invocation's named resources.
            try:
                OwnedContainers(names).close()
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
