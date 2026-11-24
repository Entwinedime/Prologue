"""Launch the application, native control and observer in their own images."""

import json
import subprocess
from contextlib import contextmanager
from time import monotonic, sleep


def inspect_image(image):
    result = subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True,
                            text=True, timeout=15)
    manifest, = json.loads(result.stdout)
    labels = manifest["Config"]["Labels"] or {}
    return {"id": manifest["Id"], "framework_revision": labels.get("org.prologue.sglang.revision")}


def mount(source, destination, *, writable=False):
    return ["--mount", f"type=bind,src={source},dst={destination}" + ("" if writable else ",readonly")]


class Deployment:
    def __init__(self, output, store, spec, clock):
        self.output = output
        self.store = store
        self.spec = spec
        self.clock = clock

    @staticmethod
    def remove(container):
        result = subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True, timeout=10)
        if result.returncode and "No such container" not in result.stderr:
            raise ValueError(result.stderr.strip())

    @contextmanager
    def start(self, role, name, options, arguments, *, socket=None, **streams):
        container = f"prologue-{self.spec['run_id']}-{name}"
        image = self.spec["images"][role]["id"]
        command = ["docker", "run", "--rm", "--pull", "never", "--name", container,
                   *options, image, *map(str, arguments)]
        # An interrupted host client can leave a container belonging to this exact run.
        self.remove(container)
        if socket is not None:
            socket.unlink(missing_ok=True)
        with subprocess.Popen(command, text=True, **streams) as process:
            try:
                yield process
            finally:
                if process.poll() is None:
                    try:
                        self.remove(container)
                    finally:
                        process.kill()
                        process.wait()

    def application_options(self, writable=False):
        return ["--runtime", "runc", "--network", "none", *mount(self.output, "/run"),
                *mount(self.store, "/library", writable=writable)]

    def app(self, arguments, *, sources=(), name="app", writable=False):
        options = self.application_options(writable)
        for index, source in enumerate(sources):
            options.extend(mount(source, f"/source/{index}"))
        return self.read("application", name, options, arguments, self.clock.remaining())

    def read(self, role, name, options, arguments, timeout):
        with self.start(role, name, options, arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            output, errors = process.communicate(timeout=timeout)
            if process.returncode:
                raise ValueError(errors.strip() or f"{arguments[0]} exited with {process.returncode}")
        return json.loads(output)

    def predict(self, target, release, directory):
        self.clock.remaining()
        socket = directory / "control.sock"
        options = self.application_options() + mount(directory, "/control", writable=True)
        arguments = ["predict", "/run/task.json", "--configuration", target["configuration"],
                     "--workload", target["workload"], "--hardware", "/run/hardware.json",
                     "--store", "/library", "--release", release, "--socket", "/control/control.sock",
                     "--framework-revision", self.spec["images"]["control"]["framework_revision"]]
        with (directory / "application.log").open("w", encoding="utf-8") as errors:
            with self.start("application", "predict", options, arguments, stdout=subprocess.PIPE,
                            stderr=errors, socket=socket) as prediction:
                self.wait_for_socket(socket, prediction)
                if socket.exists():
                    output = self.drive(prediction, directory)
                else:
                    output, _ = prediction.communicate(timeout=self.clock.remaining())
        if prediction.returncode:
            raise ValueError((directory / "application.log").read_text(encoding="utf-8").strip()
                             or f"prediction exited with {prediction.returncode}")
        return json.loads(output)

    def wait_for_socket(self, socket, prediction):
        deadline = monotonic() + self.clock.remaining()
        while prediction.poll() is None and not socket.exists():
            if monotonic() >= deadline:
                raise TimeoutError("prediction never opened the control socket")
            sleep(0.05)

    def drive(self, prediction, directory):
        options = ["--runtime", "runc", "--network", "none", *mount(directory, "/control", writable=True)]
        with (directory / "control.log").open("w", encoding="utf-8") as log:
            with self.start("control", "control", options, ["/control/control.sock"],
                            stdout=log, stderr=subprocess.STDOUT) as control:
                output, _ = prediction.communicate(timeout=self.clock.remaining())
                if (control.wait(timeout=self.clock.remaining()) and prediction.returncode == 0
                        and json.loads(output).get("status") == "complete"):
                    raise ValueError("native control exited unsuccessfully after a complete prediction")
        return output

    def observe(self, arguments, *, timeout, network="none"):
        observer = self.spec["images"]["observer"]
        if observer["framework_revision"] != self.spec["images"]["control"]["framework_revision"]:
            raise ValueError("the observer and controller use different native revisions")
        model = self.spec["model"]
        options = ["--runtime", "ascend", "--init", "--network", network, "--ipc", "host",
                   "--env", f"ASCEND_VISIBLE_DEVICES={self.spec['device']}", "--env", "OMP_NUM_THREADS=1",
                   "--env", "SGLANG_PLUGINS=prologue",
                   *mount("/usr/local/Ascend/driver", "/usr/local/Ascend/driver"),
                   *mount("/etc/ascend_install.info", "/etc/ascend_install.info"),
                   *mount("/etc/hccn.conf", "/etc/hccn.conf"),
                   *mount(self.spec["model_directory"], model), *mount(self.output, "/run", writable=True)]
        return self.read("observer", "observe", options, arguments, timeout)
