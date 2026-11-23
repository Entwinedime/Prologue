"""Connect the native decision worker to a local simulation session."""

import argparse
import json
import os
import socket
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.integrations.runtimes.pytorch.control.cpu import CpuDeclarations


class Peer:
    def __init__(self, stream):
        self.stream = stream
        self.now_ns = 0

    def call(self, operation, **fields):
        self.stream.write(json.dumps({"operation": operation, **fields}) + "\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise ConnectionError("simulation closed the control channel")
        answer = json.loads(line)
        if "failure" in answer:
            raise ValueError(answer["failure"])
        self.now_ns = answer["now_ns"]
        return answer["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket", type=Path)
    inputs = parser.parse_args()
    kernel_python = Path("/opt/prologue/sglang/sgl-kernel/python")

    with socket.socket(socket.AF_UNIX) as connection, TemporaryDirectory(prefix="prologue-control-") as temporary:
        connection.connect(str(inputs.socket))
        with connection.makefile("rw", encoding="utf-8") as stream:
            peer = Peer(stream)
            setup = peer.call("initialize")
            if setup["domain"]["framework_revision"] != os.environ["PROLOGUE_SGLANG_REVISION"]:
                raise ValueError("control framework revision differs from the source domain")
            model = Path(temporary)
            (model / "config.json").write_text(json.dumps(setup["model_config"]), encoding="utf-8")
            domain = setup["domain"]
            if domain["configuration"].get("pp", 1) > 1:
                from prologue.app.control_group import ControlGroup

                ControlGroup(peer, model, domain, kernel_python).drive()
            else:
                CpuDeclarations.install(kernel_python)

                from prologue.integrations.frameworks.sglang.control.scheduler import NativeScheduler

                NativeScheduler(peer, model, domain).drive()


if __name__ == "__main__":
    main()
