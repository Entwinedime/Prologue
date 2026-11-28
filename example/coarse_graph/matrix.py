"""Run bounded structure and performance comparisons on a frozen matrix."""

import argparse
import hashlib
import json
import re
import time
from contextlib import ExitStack
from pathlib import Path

from acceptance import Acceptance
from holdout import BlindHoldout
from prologue.common.io import write_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.recording import PublishedCapture, SessionCapture
from prologue.host.libraries.release import LibraryRelease, asset_path
from prologue.host.registry import AdapterRegistry
from prologue.host.runtime.docker import OwnedContainers
from prologue.host.runtime.limits import OperationLimit
from prologue.host.runtime.workspace import file_lock
from prologue.verification.core import ReferenceRecording
from run import TaskCommands


class MatrixRun:
    def __init__(self, root, matrix):
        self.root = root.resolve(strict=True)
        self.repository = Path(__file__).resolve().parents[2]
        self.matrix_path, self.matrix = matrix, json.loads(matrix.read_text())
        environment = json.loads((matrix.parent / "environment.json").read_text())
        self.images, self.budget = environment["images"], environment["validation_budget"]
        self.binary = (matrix.parent / "prologue").resolve(strict=True)
        self.name = self.matrix["id"]
        domains = [domain["id"] for domain in self.matrix["domains"]]
        cases = [case["id"] for domain in self.matrix["domains"] for case in domain["targets"]]
        if (not domains or not cases or len(set(domains)) != len(domains) or len(set(cases)) != len(cases)
                or any(re.fullmatch("[a-z0-9][a-z0-9-]*", name) is None for name in [self.name, *domains, *cases])):
            raise ValueError(
                "matrix needs unique domain/case identifiers containing lowercase letters, digits and hyphens"
            )
        self.commands = TaskCommands(self.repository, self.images)
        self.logs = self.root / "logs/independent"
        self.steps = self.logs / self.name
        self.steps.mkdir(parents=True, exist_ok=True)
        self.results = self.root / "results" / self.name
        self.results.mkdir(parents=True, exist_ok=True)
        self.storage = OperationLimit(self.root, self.budget["retained_bytes"])
        self.holdout = None
        if settings := environment.get("blind_acquisition"):
            self.holdout = BlindHoldout(self, matrix.parent / "source-manifest.json", settings)

    def step(self, name, command, charges, timeout, capture=None, *, exploration=None):
        container = f"prologue-matrix-{self.name}-{name.replace('/', '--')}"
        command = command[:2] + ["--name", container] + command[2:]
        path = self.steps / f"{name}.step.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["command"] != command:
                raise ValueError(f"frozen step command changed: {name}")
            if record["exit_code"] is None:
                raise RuntimeError(f"inspect the recorded process and containers before resuming: {path}")
            return record["exit_code"]

        # All investigation stages share one reservation owner; terminal failures never trigger retries.
        with file_lock(self.logs / ".budget.lock", blocking=True):
            previous = [json.loads(p.read_text()) for p in self.logs.rglob("*.step.json")]
            unavailable = [category for category, amount in charges.items()
                           if sum(row["charges"].get(category, 0) for row in previous) + amount > self.budget[category]]
            self.storage.check()
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"command": command, "charges": charges, "charge_basis": "reserved", "exit_code": None}
            if unavailable:
                record.update(charges={}, exit_code=125, error=f"allowance not available: {unavailable}")
            write_json(path, record, indent=2, replace=False)
        if unavailable:
            return 125

        print(f"{name}: starting", flush=True)
        started = time.monotonic()
        try:
            with path.with_suffix(".log").open("x") as log:
                code = self.storage.run(command, log, timeout)
        except BaseException as error:

            # A dead coordinator does not prove its native child stopped; retain the unresolved reservation.
            record["error"] = f"{type(error).__name__}: {error}"
            write_json(path, record, indent=2)
            OwnedContainers([container]).close()
            if isinstance(error, Exception):
                raise RuntimeError(f"operation has no confirmed terminal result: {path}") from error
            raise

        elapsed = time.monotonic() - started
        record.update(exit_code=code, elapsed_seconds=elapsed)
        try:
            if exploration is not None:
                if exploration.is_file():
                    usage = json.loads(exploration.read_text())["usage"]
                    native = usage["native_charged_ns"] / 1e9
                    processing = max(elapsed - native, usage["postprocessing_ns"] / 1e9)
                    record.update(charges={"card_seconds": usage["card_ns"] / 1e9,
                                           "native_attempts": usage["native_attempts"],
                                           "processing_seconds": processing},
                                  charge_basis="exploration")
            elif capture is None:
                record.update(charges={"processing_seconds": elapsed}, charge_basis="elapsed")
            else:
                owner, cards = capture
                if owner.is_file():
                    attempts = json.loads(owner.read_text())["attempts"]
                    if all(row["status"] != "running" for row in attempts):
                        native = sum(row["charged_ns"] for row in attempts) / 1e9
                        record.update(charges={"card_seconds": native * len(cards), "native_attempts": len(attempts),
                                               "processing_seconds": max(0, elapsed - native)}, charge_basis="owner")
        finally:

            # A malformed owner report cannot erase an already observed process exit or release its reservation.
            write_json(path, record, indent=2)
        print(f"{name}: exit {code}", flush=True)
        return code

    def process(self, name, command):
        allowance = self.budget["operation_seconds"]

        # Reserve coordinator shutdown too; the child's declared work allowance remains unchanged.
        return self.step(name, command, {"processing_seconds": allowance + 60}, allowance + 60)

    def sources(self, domain):
        name = domain["id"]
        release = LibraryRelease(self.root / domain["libraries"])
        checks = []
        for index, reference in enumerate(domain.get("template_observations", [])):
            recording = PublishedCapture(self.root / reference["capture"])
            production = self.root / reference["production"]
            output = self.results / "sources" / name / str(index)
            command = self.commands.command(
                "verify", "source", "--production", production, "--capture", recording.source,
                "--libraries", release.directory, "--output", output,
            )
            code = self.process(f"{name}/source-evidence-{index}", command)
            path = output / "source-evidence.json"
            checks.append({"recording": recording.attempt_id, "mapped": code == 0, "report": str(path)})

        for index, session in enumerate(domain.get("template_sessions", [])):
            recording = SessionCapture(self.root / session)
            output = self.results / "sources" / name / f"local-{index}"
            checks.append(self.local_source(f"{name}/local-source-evidence-{index}", recording, release, output))

        return {"id": name, "status": "source_evidence_checked", "libraries": str(release.directory), "checks": checks}

    def local_source(self, step, recording, release, output):
        command = self.commands.command(
            "verify", "source", "--from-session", recording.session, "--libraries", release.directory,
            "--output", output, "--binary", self.binary, "--modeling-image", self.images["modeling"],
            "--timeout", self.budget["operation_seconds"],
        )
        code = self.process(step, command)
        return {"recording": recording.attempt_id, "mapped": code == 0, "report": str(output / "source-evidence.json")}

    @staticmethod
    def selected_source(source, prediction):
        if "checks" not in source:
            return {"source_status": source["status"]}
        release = LibraryRelease(Path(source["libraries"]))
        with AdapterRegistry.bind(source=prediction, release=release) as (_, adapter):
            task = adapter.prediction(prediction, release.templates)
            if task.binding is None:
                return {"source_status": "source-unavailable", "source_failure": task.missing_workers()}

        # Only the observations used in this target's binding need an independently mapped source.
        checked = {row["recording"] for row in source["checks"] if row["mapped"]}
        missing = sorted(set(task.binding.recordings) - checked)
        return {"source_status": "source-evidence-unqualified" if missing else "ready",
                "source_recordings": list(task.binding.recordings), "unverified_recordings": missing}

    def evaluate(self, phase, domain, case, source, target, blind):
        native = json.loads((self.root / case["capture_input"]).read_text())
        prediction = json.loads((self.root / case["prediction_input"]).read_text())
        with AdapterRegistry.bind(source=native):
            if difference := ReferenceRecording.input_difference(native, prediction):
                raise ValueError(f"native and prediction inputs differ in {difference}")

        if phase == "blind":
            return self.holdout.explore(domain, case, prediction)
        if phase == "detailed":
            mode = case["detailed_capture"]
            if mode not in {"resource-trace", "service-reference"}:
                raise ValueError("a detailed reference requires a native graph observation")
            return self.physical_reference(phase, domain, case, native, case["observation"], mode)
        if phase == "light":
            references = [self.physical_reference(phase, domain, case, native, path, "latency-only")
                          for path in case["latency_observations"]]
            status = "recorded" if all(row["status"] == "recorded" for row in references) else "incomplete"
            return {"status": status, "references": references}
        if phase == "targets":
            return self.target(domain, case, native)

        if blind is not None:
            source = self.holdout.source(case, source, blind)

        # Only the structure and fixed-graph paths consume the detailed target recording.
        if phase in {"compare", "fixed"} and target["status"] == "produced":
            owner = self.root / case.get("observation", f"targets/{self.name}/{case['id']}/native")
            recording = self.reference(domain, owner, native, case.get("detailed_capture", "resource-trace"))
            if recording.directory != Path(target["native"]).resolve(strict=True):
                raise ValueError("target report refers to a different native observation")
        if phase == "compare":
            return self.compare(case, prediction, source, target)
        if phase == "fixed":
            return self.fixed(domain, case, source, target)
        return self.autonomous(domain, case, native, prediction, source)

    def physical_reference(self, phase, domain, case, native, destination, mode):
        owner = asset_path(self.root, destination)
        step = f"references/{owner.relative_to(self.root)}"

        # Reopen the step receipt even if a failed attempt left partial native artifacts.
        if not owner.exists() or (self.steps / f"{step}.step.json").is_file():
            timeout = case.get(f"{phase}_timeout_seconds", self.budget["task_seconds"])
            code = self.observe(step, domain, self.root / case["capture_input"], owner, mode, timeout)
            if code:
                return {"status": "native-capture-failed", "capture_exit": code, "native": str(owner)}

        recording = self.reference(domain, owner, native, mode)
        with AdapterRegistry.bind(source=native) as (_, adapter):
            cards = adapter.native(native, json.loads((self.root / domain["deployment"]).read_text())).platform.devices

        elapsed = NativeAttempt.usage(owner).native_ns / 1e9
        return {"status": "recorded", "native": str(recording.directory), "attempt": recording.attempt_id,
                "card_seconds": elapsed * len(cards)}

    def reference(self, domain, owner, native, mode):
        deployment = json.loads((self.root / domain["deployment"]).read_text())
        conditions = {"image": self.images["source"], "deployment": deployment, mode.replace("-", "_"): True}
        if mode == "latency-only":
            conditions["host_timing"] = None
        return PublishedCapture(owner, native, conditions=conditions)

    def target(self, domain, case, capture_input):
        name = case["id"]
        result = {"id": name}
        mode = case.get("detailed_capture", "resource-trace")
        if mode not in {"resource-trace", "service-reference"}:
            raise ValueError("a structure target needs a detailed native observation")

        if "observation" in case:

            # An explicit immutable task replaces collection; the same complete input check still precedes production.
            output = (self.root / case["observation"]).resolve(strict=True)
            result["observation"] = str(output)
        else:
            output = self.root / "targets" / self.name / name / "native"
            task = self.root / case["capture_input"]
            timeout = case.get("detailed_timeout_seconds", self.budget["task_seconds"])
            code = self.observe(f"{name}/capture", domain, task, output, mode, timeout)
            result["capture_exit"] = code
            if code:
                return result | {"status": "native-capture-failed"}

        recording = self.reference(domain, output, capture_input, mode)

        # A physical target has one observed graph across all source modes in this frozen matrix.
        identity = hashlib.sha256(f"{recording.directory}\0{recording.attempt_id}".encode()).hexdigest()
        step = f"graphs/{identity}"
        production = self.root / "targets" / self.name / step
        command = self.commands.command(
            "capture", "project",
            "--capture", recording.directory,
            "--product", "graphs",
            "--output", production,
            "--binary", self.binary,
            "--source-image", self.images["source"],
            "--modeling-image", self.images["modeling"],
            "--budget-seconds", self.budget["operation_seconds"],
            "--keep-intermediates",
        )

        code = self.process(step, command)
        result.update(native=str(recording.directory), production=str(production), production_exit=code)
        return result | {"status": "target-production-failed" if code else "produced"}

    def observe(self, step, domain, task, output, mode, timeout):
        native = json.loads(task.read_text())
        deployment = self.root / domain["deployment"]
        with AdapterRegistry.bind(source=native) as (_, adapter):
            cards = adapter.native(native, json.loads(deployment.read_text())).platform.devices

        command = self.commands.command(
            "capture",
            "run",
            "--input",
            task,
            "--deployment",
            deployment,
            "--output",
            output,
            "--source-image",
            self.images["source"],
            "--timeout",
            timeout,
            "--budget-seconds",
            timeout,
            "--attempt-limit",
            1,
            f"--{mode}",
            model=native["model"]["path"],
        )
        allowance = timeout + 60
        charges = {"card_seconds": allowance * len(cards), "native_attempts": 1, "processing_seconds": 60}
        return self.step(step, command, charges, allowance, (output / "execution.json", cards))

    def compare(self, case, prediction, source, target):
        name = case["id"]
        result = {"id": name, "target_status": target["status"]}
        if target["status"] != "produced":
            return result | {"status": "target-prerequisite-unqualified"}
        result.update(self.selected_source(source, prediction))
        if result["source_status"] != "ready":
            return result | {"status": "source-prerequisite-unqualified"}

        output = self.results / "cases" / name
        timing = ("--peer-handoffs",) if case.get("peer_handoffs") else ()
        command = self.commands.command(
            "verify", "structure",
            "--capture",
            target["native"],
            "--reference",
            target["production"],
            "--libraries",
            source["libraries"],
            "--input",
            self.root / case["prediction_input"],
            "--output",
            output,
            "--binary",
            self.binary,
            "--modeling-image",
            self.images["modeling"],
            "--control-image",
            self.images["control"],
            "--timeout",
            self.budget["task_seconds"],
            *timing,
        )

        # One process holds the checked native case through extraction, prediction and comparison.
        allowance = 3 * self.budget["operation_seconds"] + 2 * self.budget["task_seconds"] + 60
        result["verification_exit"] = self.step(f"{name}/verify", command, {"processing_seconds": allowance}, allowance)
        path = output / "result.json"
        if not path.exists():
            return result | {"status": "verification-error"}
        return result | json.loads(path.read_text()) | {"report": str(path)}

    def fixed(self, domain, case, source, target):
        name = case["id"]
        result = {"id": name}
        if case.get("detailed_capture") != "service-reference":
            return result | {"status": "not-selected"}
        if target["status"] != "produced":
            return result | {"status": "target-prerequisite-unqualified"}
        release = LibraryRelease(Path(source.get("libraries", self.root / domain["libraries"])))
        if release.costs is None:
            return result | {"status": "cost-asset-unavailable"}

        evidence = ("--capture", target["native"], "--reference", target["production"], "--costs", release.costs)
        if groups := case.get("fixed_work_groups"):
            evidence += ("--work-groups", self.root / groups)
        return result | self.verify_performance(name, "fixed", "fixed-graph", evidence)

    def autonomous(self, domain, case, native, prediction, source):
        name = case["id"]
        explicit = "latency_observations" in case
        paths = case.get("latency_observations", [f"targets/{self.name}/{name}/latency"])
        result = {"id": name, **self.selected_source(source, prediction)}
        references = []
        for index, path in enumerate(paths):
            owner = asset_path(self.root, path)
            step = f"{name}/{index + 1}" if explicit else name
            try:
                references.append(self.autonomous_reference(step, domain, case, native, source, result, owner,
                                                            explicit))
            except (OSError, ValueError, KeyError) as error:
                references.append({"status": "artifact-error", "native_task": str(owner), "error": str(error)})

        if not explicit:
            return result | references[0]
        status = "compared" if all(row["status"] == "compared" for row in references) else "incomplete"
        return result | {"status": status, "references": references}

    def autonomous_reference(self, step, domain, case, native, source, selection, owner, explicit):
        result = {"native_task": str(owner)}
        if not explicit:
            timeout = case.get("latency_timeout_seconds", self.budget["task_seconds"])
            code = self.observe(f"{step}/latency", domain, self.root / case["capture_input"], owner, "latency-only",
                                timeout)
            if code:
                return result | {"status": "native-capture-failed", "capture_exit": code}
        self.reference(domain, owner, native, "latency-only")
        if selection["source_status"] != "ready":
            return result | {"status": "source-prerequisite-unqualified"}

        evidence = ("--capture", owner, "--libraries", source["libraries"],
                    "--input", self.root / case["prediction_input"], "--control-image", self.images["control"])
        if case.get("retain_execution_trace"):
            evidence += ("--details",)
        return result | self.verify_performance(step, "autonomous", "autonomous", evidence)

    def verify_performance(self, name, phase, operation, evidence):
        output = self.results / phase / name
        timeout = self.budget["operation_seconds"]
        command = self.commands.command(
            "verify", operation, *evidence, "--output", output,
            "--binary", self.binary, "--modeling-image", self.images["modeling"], "--timeout", timeout,
        )
        allowance = (2 * timeout if phase == "fixed" else timeout) + 60
        code = self.step(f"{name}/{phase}", command, {"processing_seconds": allowance}, allowance)
        path = output / "result.json"
        if not path.exists():
            return {"status": "verification-error", "verification_exit": code}
        return {"status": json.loads(path.read_text())["status"], "report": str(path)}

    def run(self, phase, domain_name):
        if self.matrix["status"] != "frozen":
            raise ValueError("matrix is not frozen; no execution started")
        domains = [row for row in self.matrix["domains"] if domain_name is None or row["id"] == domain_name]
        if not domains:
            raise ValueError("matrix contains no selected domain")

        report = "comparisons" if phase == "compare" else phase
        output = self.results / f"{report}.json"
        with ExitStack() as contexts:
            contexts.enter_context(file_lock(self.steps / f".{phase}.lock"))
            blind = self.blind_gate(phase, contexts)
            if phase == "summary":
                summary = Acceptance(self.holdout).report()
                write_json(self.results / "acceptance-summary.json", summary, indent=2)
                return summary["status"] == "passed"

            results = {}
            for name in {"sources", "targets", report}:
                path = self.results / f"{name}.json"
                results[name] = json.loads(path.read_text()) if path.exists() else {}
            sources, targets, records = (results[name] for name in ("sources", "targets", report))
            needs_native = (
                phase in {"blind", "detailed", "light"}
                or phase == "autonomous"
                and any("latency_observations" not in case for domain in domains for case in domain["targets"])
                or phase == "targets"
                and any("observation" not in case for domain in domains for case in domain["targets"])
            )
            if needs_native:

                # Only actual collection occupies the cards; explicit observation reuse stays in the CPU lane.
                contexts.enter_context(file_lock(self.logs / ".native.lock"))
            for domain in domains:
                for case in [domain] if phase == "sources" else domain["targets"]:
                    try:
                        if phase == "sources":
                            result = self.sources(domain)
                        else:
                            source = sources.get(domain["id"], {"status": "not-executed"})
                            target = targets.get(case["id"], {"status": "not-executed"})
                            result = self.evaluate(phase, domain, case, source, target, blind.get(case["id"]))
                    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                        # Bad artifacts have a terminal finding; unresolved processes still stop above.
                        result = {
                            "id": case["id"],
                            "status": "artifact-error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    records[case["id"]] = result
                    write_json(output, records, indent=2)
        return True

    def blind_gate(self, phase, contexts):
        if self.holdout is None:
            if phase in {"blind", "summary"}:
                raise ValueError("this matrix has no frozen blind acquisition settings")
            return {}

        # All blind and reference phases share the same gate, including when called by separate coordinators.
        contexts.enter_context(file_lock(self.steps / ".holdout.lock"))
        self.holdout.check_freeze(self.matrix_path)
        if phase == "blind":
            self.holdout.require_no_references()
        elif phase != "sources":
            return self.holdout.require_blind_results(before_collection=phase in {"detailed", "light"})
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing investigation directory")
    parser.add_argument("--matrix", type=Path, required=True, help="frozen matrix beside environment.json and prologue")
    parser.add_argument(
        "--phase",
        choices=("sources", "blind", "detailed", "light", "targets", "compare", "fixed", "autonomous", "summary"),
        required=True,
    )
    parser.add_argument("--domain", help="one named domain; omit for the whole stage")
    arguments = parser.parse_args()
    raise SystemExit(0 if MatrixRun(arguments.root, arguments.matrix).run(arguments.phase, arguments.domain) else 1)


if __name__ == "__main__":
    main()
