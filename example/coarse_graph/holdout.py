"""Keep each blind exploration isolated from all full-service targets in its matrix."""

import shutil
from hashlib import sha256

from prologue.common.io import read_json, write_json
from prologue.host.acquisition.recording import SessionCapture
from prologue.host.libraries.release import LibraryRelease, asset_path
from prologue.host.registry import AdapterRegistry
from prologue.host.runtime.workspace import TaskWorkspace, file_lock


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            checksum.update(block)

    return checksum.hexdigest()


class BlindHoldout:
    def __init__(self, matrix, manifest, allowances):
        self.matrix, self.manifest = matrix, manifest
        self.allowances = allowances
        self.results = matrix.results
        self.root = matrix.root
        self.domains = matrix.matrix["domains"]

    def check_freeze(self, path):
        frozen = read_json(self.manifest)
        environment = self.manifest.parent / "environment.json"
        specification = self.manifest.parent / "specification.json"
        if (frozen["images"] != self.matrix.images or digest(path) != frozen["matrix"]
                or digest(environment) != frozen["environment"] or digest(self.matrix.binary) != frozen["binary"]
                or digest(specification) != frozen["specification"]):
            raise ValueError("matrix, executable or image pins differ from the frozen manifest")

        for relative, given in frozen["source"].items():
            if digest(asset_path(self.matrix.repository, relative)) != given:
                raise ValueError(f"source changed after freeze: {relative}")

        for relative, given in frozen["deployment"].items():
            if digest(asset_path(self.manifest.parent / "deployments", relative)) != given:
                raise ValueError(f"deployment changed after freeze: {relative}")

        for directory in {asset_path(self.root, domain["libraries"]) for domain in self.domains}:
            files = {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()}
            if files != frozen["libraries"][directory.name]:
                raise ValueError(f"initial release changed after freeze: {directory}")

        seal = read_json(self.results / "draw-integrity.json")
        draw = self.root / "inputs/independent" / self.matrix.name / "draw.json"
        if (digest(self.manifest) != seal["freeze_sha256"] or digest(draw) != seal["draw_sha256"]
                or read_json(draw)["seed"] != seal["freeze_sha256"]):
            raise ValueError("request draw must be derived from the frozen manifest")

        inputs = {case[key] for case in self.coordinates() for key in ("capture_input", "prediction_input")}
        if inputs != seal["inputs"].keys() or any(
            digest(asset_path(self.root, name)) != value for name, value in seal["inputs"].items()
        ):
            raise ValueError("matrix inputs differ from the sealed request draw")

    def coordinates(self):
        return (case for domain in self.domains for case in domain["targets"])

    def require_no_references(self):
        for case in self.coordinates():
            for field in ("observation", "latency_observation"):
                if asset_path(self.root, case[field]).exists():
                    raise ValueError("blind acquisition must finish before any full-service reference exists")

    def require_blind_results(self):
        path = self.results / "blind.json"
        if not path.is_file():
            raise ValueError("full-service reference collection requires the completed blind phase")

        records = read_json(path)
        if any("report" not in records.get(case["id"], {}) for case in self.coordinates()):
            raise ValueError("every target needs a terminal formal exploration before reference collection")
        return records

    @staticmethod
    def seed_store(initial, store):
        identity = initial.identity
        if not identity or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("blind acquisition requires a frozen published baseline")

        pointer = {"release": f"releases/{identity}"}
        with file_lock(store / ".lock"):
            with TaskWorkspace(store).initialize("current.json") as draft:
                if draft is not None:
                    destination = draft / pointer["release"]
                    shutil.copytree(initial.directory, destination)
                    if not (destination / "evidence.json").exists():
                        write_json(destination / "evidence.json", {}, replace=False)

                    write_json(draft / "current.json", pointer, replace=False)

    def explore(self, domain, case, source):
        name = case["id"]
        directory = self.results / "blind" / name
        output = directory / "exploration"
        library = directory / "libraries"
        initial = LibraryRelease(asset_path(self.root, domain["libraries"]))
        self.seed_store(initial, library)

        input_path = directory / "exploration-input.json"
        with AdapterRegistry.bind(source=source, release=initial) as (route, _):
            framework = route.framework

        study = {"framework": framework, "model": source["model"],
                 "configurations": [{"id": case["configuration"], "instances": source["instances"]}],
                 "workloads": [source["workload"]]}
        if not input_path.exists():
            write_json(input_path, study, indent=2, replace=False)
        if read_json(input_path) != study:
            raise ValueError("blind exploration input changed after the first launch")

        result = self.predict(name, source, domain, input_path, output, library)
        report = read_json(output / "result.json")
        if result not in {0, 1} or (result == 0) != (report["status"] == "complete"):
            raise ValueError("blind exploration has no consistent terminal result")

        release = LibraryRelease(library)
        if str(release.directory) != report["libraries"]:
            raise ValueError("blind exploration did not pin its final library release")

        first = output / "targets/0/attempts/0"
        warm = (first / "result.json").is_file()
        entry = {"id": name, "status": report["status"], "report": str(output / "result.json"),
                 "initial_release": initial.identity, "final_release": release.identity,
                 "libraries": report["libraries"],
                 "initial_hot": warm, "usage": report["usage"], "sessions": str(output / "acquisition/sessions")}
        for gap in ("template-gap", "cost-gap"):
            if (path := first / f"{gap}.json").is_file():
                entry["initial_gap"] = {"kind": gap, "status": read_json(path)["status"], "report": str(path)}

        return entry

    def predict(self, name, source, domain, input_path, output, library):
        output.mkdir(parents=True, exist_ok=True)
        deployment = asset_path(self.root, domain["deployment"])
        binary = self.matrix.binary
        images = self.matrix.images
        limits = self.allowances
        files = [(input_path, True), (deployment, True), (binary, True), (output, False), (library, False)]
        command = self.matrix.commands.command(
            "predict", "explore", "--input", input_path, "--libraries", library, "--output", output,
            "--binary", binary, "--deployment", deployment, "--control-image", images["control"],
            "--modeling-image", images["modeling"], "--source-image", images["source"], "--timeout",
            limits["timeout_seconds"], "--card-budget-seconds", limits["card_seconds"], "--attempt-limit",
            limits["native_attempts"], "--postprocessing-budget-seconds", limits["processing_seconds"],
            "--storage-bytes", limits["storage_bytes"], model=source["model"]["path"], files=files,
        )
        charges = {key: limits[key] for key in ("card_seconds", "processing_seconds", "native_attempts")}

        # The formal task accounts for probes and fitting. The coordinator also owns prediction and shutdown time.
        charges["processing_seconds"] += limits["prediction_seconds"]
        return self.matrix.step(f"{name}/blind", command, charges, sum(
            charges[key] for key in ("card_seconds", "processing_seconds")
        ), exploration=output / "result.json")

    def source(self, case, source, blind):
        release = LibraryRelease(self.root / blind["libraries"])
        if release.identity != blind["final_release"]:
            raise ValueError("blind result no longer selects the sealed release")

        known = {row["recording"] for row in source.get("checks", ()) if row["mapped"]}
        missing = set(release.templates.observations) - known
        checks = list(source.get("checks", ()))
        sessions = self.root / blind["sessions"]
        for index, path in enumerate(sorted(sessions.glob("*/session/result.json"))):
            receipt = read_json(path)
            if receipt.get("native_recording") not in missing:
                continue

            recording = SessionCapture(path.parent, receipt)
            name = case["id"]
            output = self.results / "sources" / name / f"blind-{index}"
            checks.append(self.matrix.local_source(f"{name}/blind-source-{index}", recording, release, output))

        return source | {"libraries": str(release.directory), "checks": checks}
