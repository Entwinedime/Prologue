"""Send scheduler action cuts and virtual observations through the control peer."""

from contextlib import contextmanager

from prologue.integrations.frameworks.sglang.control.requests import PublishedRequests
from prologue.integrations.frameworks.sglang.shared.batches import (
    DecisionClocks,
    RequestSubmissions,
)
from prologue.integrations.frameworks.sglang.shared.calls import CallSegments
from prologue.integrations.frameworks.sglang.shared.configuration import ParallelGroups
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import DisaggregatedCalls
from prologue.integrations.frameworks.sglang.shared.metadata_calls import HostReads, MetadataCollectives
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.resources import ResourceCalls
from prologue.integrations.frameworks.sglang.shared.selection import BatchSelection


class FlowExecution(CallSegments):
    """Expose native call continuations to the simulator while preserving the scheduler's control-flow position."""

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.started = False
        self.pending_inputs = set(range(len(scheduler.request_input.submissions.requests)))
        self.request_state = scheduler.request_state
        self.rank = scheduler.graph_rank
        self.groups = ParallelGroups()
        self.clocks = DecisionClocks(self)
        self.clocks.install()
        HostReads(self).install()
        self.selection = BatchSelection(self)
        self.selection.install()
        self.resources = ResourceCalls(self)
        self.resources.install()
        self.resources.bind(
            scheduler.tree_cache, scheduler.token_to_kv_pool_allocator, scheduler.req_to_token_pool, native=False,
        )
        scheduler.output_streamer.send_to_detokenizer = PublishedRequests(
            self, scheduler.output_streamer.send_to_detokenizer.socket is not None,
        )

        if scheduler.server_args.disaggregation_mode != "null":
            DisaggregatedCalls(self).install(scheduler.server_args.disaggregation_mode)

    @property
    def enabled(self):
        return self.started

    def input(self, request):
        self.pending_inputs.remove(RequestSubmissions.index(request.rid))
        self.started = True

    def drained(self):

        # Close at the same local idle boundary as capture, never between future arrivals or inside a call.
        if not self.pending_inputs and self.current.get() is None:
            self.started = False

    def request_id(self, native_id):
        return self.scheduler.request_input.request_id(native_id)

    def work(self, batch, members=None, *, phase="prefill"):
        return self.scheduler.batch_names.describe(batch) | super().work(batch, members, phase=phase)

    @staticmethod
    def completed(requests):
        return [
            {"request": RequestSubmissions.index(req.rid), "generated_tokens": len(req.output_ids),
             "finished": req.finished(), "emits_output": False}
            for req in requests
        ]

    def collective(self, original, boundary, arguments):

        # Inputs are available at graph entry; consuming a result must not wait for peers' graph returns.
        work = original(**{**arguments, "async_op": True})
        returned = self.scheduler.action(
            boundary.operation, boundary.parameters, boundary.work, condition=boundary.condition
        )

        # Native Work publishes into the tensor at wait; snapshot only when execution tracing requests evidence.
        values, outputs = (
            MetadataCollectives.snapshot(arguments, boundary.condition["collective"]["root"])
            if "observation" in returned else (None, None)
        )
        work.wait()

        if values is not None:
            values["output"] = outputs.flatten().tolist()
            self.scheduler.peer.call("action_values", boundary=returned["observation"], values={"collective": values})

    def device_collective(self, original, arguments):
        frame = self.current.get()
        if frame.operation != "batch_sync_exchange":
            return super().device_collective(original, arguments)

        # This native method has one exchange and private tensor postprocessing; the whole declared graph owns both.
        work = original(**{**arguments, "async_op": True})
        self.suspend(frame, "call.return")
        work.wait()

    def clock_sample(self, original, query):
        time_ns = self.scheduler.peer.call("clock") if query is None else query["time_ns"]

        # Native policy uses zero as an unset timestamp; a shared positive epoch preserves elapsed durations.
        ticks = time_ns + self.scheduler.clock_epoch_ns
        return ticks / 1_000_000_000, ticks

    @contextmanager
    def read_boundary(self, kind, original, work, condition):

        # Native nonblocking reads complete at this virtual instant, independently of optional tracing.
        returned = self.scheduler.action("query", {}, work, condition=condition)
        query = {"boundary": returned["read_boundary"], "time_ns": returned["completed_ns"]}
        yield query

        self.scheduler.peer.call(
            "read_result", boundary=query["boundary"], result=query["result"], clocks=query["clocks"],
        )

    @contextmanager
    def boundary(self, original, boundary):
        with self.request_state.record() as accesses:
            yield
        self.scheduler.bind_state(boundary.work, accesses)
        self.scheduler.action(
            boundary.operation, boundary.parameters, boundary.work, condition=boundary.condition,
            tensor_inputs={name: tensor for name, (tensor, _) in boundary.inputs.items()},
            tensor_outputs={name: tensor for name, (tensor, _) in boundary.outputs.items()},
        )

    @contextmanager
    def enclosing(self, frame):
        yield
        if frame.values:
            self.scheduler.peer.call(
                "action_values", boundary=frame.observations[-1],
                values={**frame.values, "segments": frame.observations},
            )

    @contextmanager
    def fragment(self, frame):
        with self.request_state.record() as accesses:
            yield

        # Submission facts belong to this segment, not to the caller context inherited by later children.
        work = {**frame.work, "continuation": frame.continuation}
        self.scheduler.bind_state(work, accesses)
        condition = {**frame.observed_condition, "owner": frame.operation}
        if not frame.host_fragment:
            condition.update(entry=frame.entry, exit=frame.exit)

        outputs = frame.outputs if frame.terminal else ()
        tensors = {name: tensor for name, (tensor, _) in frame.tensor_outputs.items()}
        inputs = {name: tensor for name, (tensor, _) in {**frame.tensor_inputs, **frame.metadata_inputs}.items()}
        returned = self.scheduler.action(
            "segment",
            frame.parameters,
            work,
            outputs,
            condition=condition,
            tensor_inputs=inputs,
            tensor_outputs=tensors,
            allocations=frame.tensor_allocations,
        )

        if self.request_state.observe:
            frame.observations.append(returned["observation"])
        if accesses.values:
            self.scheduler.peer.call("action_values", boundary=returned["observation"], values={"state": [
                {**value, "request": self.request_id(value["request"])} for value in accesses.values
            ]})

    def process(self, original, batch, result):
        parameters, work = self.scheduler.description(batch)
        if self.scheduler.ps.pp_size > 1:
            work["observation_context"] = {"pipeline": PipelineABI.control_context(self)}

        with self.call(
            "result", original, parameters=parameters, work=work, host_inputs=HostReads.result_inputs(result)
        ) as frame:
            original(batch, result)

            # Prebuilt has its own result boundary, also used directly during native Decode selection.
            if not batch.forward_mode.is_prebuilt():
                frame.outputs = self.completed(batch.reqs)
