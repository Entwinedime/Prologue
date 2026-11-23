"""Multiplex native actor processes through one simulated control session."""

import multiprocessing
import traceback
from multiprocessing.connection import wait

from prologue.integrations.runtimes.pytorch.control.cpu import CpuDeclarations


class ActorPeer:
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


def run_actor(connection, model, domain, identity, kernel_python):
    CpuDeclarations.install(kernel_python)

    from sglang.srt.distributed import parallel_state

    from prologue.integrations.frameworks.sglang.control.scheduler import Finished

    if isinstance(identity, str):
        from prologue.integrations.frameworks.sglang.control.disaggregation import DisaggregatedScheduler

        scheduler_type = DisaggregatedScheduler
    else:
        from prologue.integrations.frameworks.sglang.control.pipeline import PipelineScheduler

        scheduler_type = PipelineScheduler

    with connection:
        peer = ActorPeer(connection)
        try:
            scheduler = scheduler_type(peer, model, domain, identity)
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
    def __init__(self, peer, model, domain, kernel_python, identities):
        self.peer = peer
        self.model = model
        self.domain = domain
        self.kernel_python = kernel_python
        self.identities = identities
        self.processes = []
        self.actors = []

    def drive(self):
        context = multiprocessing.get_context("spawn")
        try:
            for identity in self.identities:
                parent, child = context.Pipe()
                process = context.Process(target=run_actor, args=(
                    child, self.model, self.domain, identity, self.kernel_python,
                ), name=f"native-{identity}")
                self.actors.append(parent)
                with child:
                    process.start()
                self.processes.append(process)
            self.initialize()
            self.dispatch()
            for process in self.processes:
                process.join()
                if process.exitcode:
                    raise ValueError(f"{process.name} exited with status {process.exitcode}")
        finally:
            for connection in self.actors:
                connection.close()
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join()

    def initialize(self):
        awaiting = {connection: actor for actor, connection in enumerate(self.actors)}
        while awaiting:
            for connection in wait(awaiting):
                message = self.receive(awaiting[connection])
                if message["operation"] == "ready":
                    del awaiting[connection]
                    continue
                result = self.peer.call("setup_actor", actor=awaiting[connection], call=message)
                connection.send({"now_ns": self.peer.now_ns, "result": result, "done": False})
        for connection in self.actors:
            connection.send({"now_ns": self.peer.now_ns, "result": None, "done": False})

    def dispatch(self):
        ready = range(len(self.actors))
        while True:
            calls = [{"actor": actor, "call": self.receive(actor)} for actor in ready]
            reply = self.peer.call("turn", calls=calls)
            if reply["done"]:
                for connection in self.actors:
                    connection.send({"done": True})
                return
            actor = reply.pop("actor")
            self.actors[actor].send(reply)
            ready = (actor,)

    def receive(self, actor):
        try:
            message = self.actors[actor].recv()
        except EOFError as error:
            process = self.processes[actor]
            process.join(timeout=1)
            raise ConnectionError(f"{process.name} disconnected with exit status {process.exitcode}") from error
        if "failure" in message:
            raise RuntimeError(f"{self.processes[actor].name} failed:\n{message['failure']}")
        return message
