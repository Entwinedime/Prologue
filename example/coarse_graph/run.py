"""Launch the formal exploration task with the repository example inputs."""

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class TaskCommands:
    """Call a task module from a coordinator sharing the host daemon's paths."""

    def __init__(self, repository, images):
        self.repository, self.images = repository, images

    def command(self, *arguments, model=None, files=None):

        # Independent experiments need their workspace. Prediction receives only its inputs and result directory.
        if files is None:
            mounts = [(self.repository, self.repository, False)]
        else:
            source = self.repository / "src"
            catalog = self.repository / "shared"
            mounts = [(source, source, True), (catalog, catalog, True), *((path, path, readonly) for path, readonly in files)]
        if model is not None:
            destination = Path(model).absolute()
            mounts.append((destination.resolve(strict=True), destination, True))
            from prologue.integrations.platforms.ascend.host.deployment import AscendDeployment

            for name in AscendDeployment.driver_mounts:
                mounts.append((Path(name).resolve(strict=True), Path(name), True))

        mounts.extend([
            (Path("/var/run/docker.sock"), Path("/var/run/docker.sock"), False),
            (Path(shutil.which("docker")).resolve(strict=True), Path("/usr/bin/docker"), True),
        ])
        command = ["docker", "run", "--rm", "--init", "-i", "--runtime", "runc", "--network", "none"]
        for source, destination, readonly in mounts:
            command.extend(["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")])
        command.extend([
            "--workdir", str(self.repository),
            "--env", "PYTHONPATH=src",
            "--entrypoint", "python3", self.images["control"], "-m", "prologue",
        ])
        return command + [str(value) for value in arguments]


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input.json"))
    parser.add_argument("--libraries", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path, default=repository / "build/coarse-graph/prologue")
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")
    parser.add_argument("--control-image", default="prologue-control:ubuntu22.04")
    options, remaining = parser.parse_known_args()
    images = {"control": options.control_image, "modeling": options.modeling_image}
    commands = TaskCommands(repository, images)
    if "--help" in remaining or "-h" in remaining:
        raise SystemExit(subprocess.call(commands.command("predict", "explore", "--help", files=())))
    if options.libraries is None:
        parser.error("--libraries is required")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (options.output or repository / "data" / f"{stamp}_exploration").resolve()
    libraries = options.libraries.resolve()
    source, binary = options.input.resolve(strict=True), options.binary.resolve(strict=True)
    if output.is_relative_to(libraries) or libraries.is_relative_to(output):
        parser.error("exploration output and published libraries must be separate")
    output.mkdir(parents=True, exist_ok=True)
    files = [(source, True), (binary, True), (output, False)]
    if libraries.exists():
        files.append((libraries, True))

    command = commands.command(
        "predict", "explore", "--input", source, "--libraries", libraries, "--output", output, "--binary", binary,
        "--modeling-image", images["modeling"], "--control-image", images["control"], *remaining, files=files,
    )
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
