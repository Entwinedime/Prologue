"""Multiplex native stage processes through one simulated control session."""

import multiprocessing
import traceback
from multiprocessing.connection import wait

from prologue.integrations.runtimes.pytorch.control.cpu import CpuDeclarations


class StagePeer:
    def __init__(self, connection):
        self.connection = connection
        self.now_ns = 0

    def call(self, operation, **fields):
        from prologue.integrations.frameworks.sglang.control.scheduler import Finished

        self.connection.send({"operation": operation, **fields})
        answer = self.connection.recv()
        if answer["done"]:
            raise Finished
        self.now_ns = answer["now_ns"]
        return answer["result"]


def run_stage(connection, model, domain, stage, kernel_python):
    CpuDeclarations.install(kernel_python)

    from sglang.srt.distributed import parallel_state

    from prologue.integrations.frameworks.sglang.control.pipeline import PipelineScheduler
    from prologue.integrations.frameworks.sglang.control.scheduler import Finished

    with connection:
        peer = StagePeer(connection)
        try:
            scheduler = PipelineScheduler(peer, model, domain, stage)
            peer.call("ready")
            scheduler.drive()
        except Finished:
            pass
        except Exception:
            connection.send({"failure": traceback.format_exc()})
            raise
        finally:
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()


class ControlGroup:
    def __init__(self, peer, model, domain, kernel_python):
        self.peer = peer
        self.model = model
        self.domain = domain
        self.kernel_python = kernel_python
        self.processes = []
        self.stages = []

    def drive(self):
        context = multiprocessing.get_context("spawn")
        width = self.domain["configuration"]["pp"]
        try:
            for stage in range(width):
                parent, child = context.Pipe()
                process = context.Process(target=run_stage, args=(
                    child, self.model, self.domain, stage, self.kernel_python,
                ))
                self.stages.append(parent)
                with child:
                    process.start()
                self.processes.append(process)
            self.initialize()
            self.dispatch()
            for process in self.processes:
                process.join()
                if process.exitcode:
                    raise ValueError(f"native stage {process.name} exited with status {process.exitcode}")
        finally:
            for connection in self.stages:
                connection.close()
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join()

    def initialize(self):
        awaiting = {connection: stage for stage, connection in enumerate(self.stages)}
        while awaiting:
            for connection in wait(awaiting):
                message = self.receive(awaiting[connection])
                if message["operation"] == "ready":
                    del awaiting[connection]
                    continue
                result = self.peer.call("setup_actor", actor=awaiting[connection], call=message)
                connection.send({"now_ns": self.peer.now_ns, "result": result, "done": False})
        for connection in self.stages:
            connection.send({"now_ns": self.peer.now_ns, "result": None, "done": False})

    def dispatch(self):
        ready = range(len(self.stages))
        while True:
            calls = [{"actor": stage, "call": self.receive(stage)} for stage in ready]
            reply = self.peer.call("turn", calls=calls)
            if reply["done"]:
                for connection in self.stages:
                    connection.send({"done": True})
                return
            stage = reply.pop("actor")
            self.stages[stage].send(reply)
            ready = (stage,)

    def receive(self, stage):
        try:
            message = self.stages[stage].recv()
        except EOFError as error:
            process = self.processes[stage]
            process.join(timeout=1)
            raise ConnectionError(f"native stage {stage} disconnected with exit status {process.exitcode}") from error
        if "failure" in message:
            raise RuntimeError(f"native stage {stage} failed:\n{message['failure']}")
        return message
