"""Drive the native scheduler from core arrivals, pipeline signals and work boundaries."""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from weakref import WeakValueDictionary

import torch
import zmq
from torch.utils.weak import WeakIdKeyDictionary

from prologue.common.workload import ExpertRoutes, RoutedContent
from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.control.resources import AllocationDemand, MetadataRunner, PrefixCache
from prologue.integrations.frameworks.sglang.shared.batches import (
    BatchTensors,
    DecisionClocks,
    RequestStateCalls,
    RequestSubmissions,
    RoutedCacheKeys,
)
from prologue.integrations.frameworks.sglang.shared.calls import (
    CallSegments,
    HostReads,
    MetadataCollectives,
    PipelineABI,
    PipelineMessages,
)
from prologue.integrations.frameworks.sglang.shared.configuration import ExpertBatch, ParallelGroups
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import DisaggregatedCalls
from prologue.integrations.frameworks.sglang.shared.resources import ResourceCalls
from prologue.integrations.frameworks.sglang.shared.selection import BatchSelection
from prologue.integrations.runtimes.pytorch.control import TensorValues


@dataclass
class ControlSignal:
    identity: str
    context: dict


class PipelineSignals:
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        self.live = WeakValueDictionary()
        self.sequence = 0
        self.enclosing = 0

    @contextmanager
    def producing(self):
        self.enclosing += 1
        try:
            yield
        finally:
            self.enclosing -= 1

    def record(self, original, event, stream=None):
        result = original(event, stream)
        if not self.scheduler.calls.enabled:
            return result
        work = self.scheduler.pipeline_messages.signal_work(stream)
        identity = self.publish(event, work["context"])
        if not self.enclosing:
            work["signals"] = {"completion": identity}
            self.scheduler.action("signal_record", {}, work, condition={"stream": work["context"]["record_stream"]})
        return result

    def publish(self, event, context):
        # Re-recording a handle creates a fresh generation; old waits retain their already submitted graph edge.
        identity = f"{self.scheduler.group}/signal/{self.sequence}"
        self.sequence += 1
        signal = ControlSignal(identity, context)
        event._prologue_signal = signal

        # Observe Python signal lifetimes, so the core retains completions only while native events own them.
        self.live[identity] = signal
        return identity

    def wait(self, event, *, device, stream=None):
        signal = event._prologue_signal
        work = self.scheduler.pipeline_messages.wait_work(signal.context, device=device, stream=stream)
        work["signals"] = {"completion": signal.identity}
        condition = {"device_wait": device}
        if device:
            condition["stream"] = work["context"]["wait_stream"]
        self.scheduler.action("signal_wait", {}, work, condition=condition)

    def wait_stream(self, original, stream, other):
        result = original(stream, other)
        if self.scheduler.calls.enabled and not self.enclosing:
            streams = PipelineABI.streams(self.scheduler, stream, other)
            work = self.scheduler.pipeline_messages.work()
            work["context"]["streams"] = streams
            self.scheduler.action("stream_wait", {}, work, condition={"streams": streams})
        return result

    def query(self, original, event):
        signal = getattr(event, "_prologue_signal", None)
        if not self.scheduler.calls.enabled or signal is None:
            return original(event)
        work = self.scheduler.pipeline_messages.work()
        work["context"]["producer"] = signal.context
        with self.scheduler.calls.read("event", original, work) as query:
            result = self.scheduler.peer.call("completed", signal=signal.identity, boundary=query["boundary"])
            query["result"] = result
            return result

    def synchronize(self, original, event):
        if not self.scheduler.calls.enabled or not hasattr(event, "_prologue_signal"):
            return original(event)
        self.wait(event, device=False)

    def wait_event(self, original, stream, event):
        if not self.scheduler.calls.enabled or not hasattr(event, "_prologue_signal"):
            return original(stream, event)
        self.wait(event, device=True, stream=stream)

    def install(self):
        HookTable.around("torch.cpu.Event.query", self.query)
        HookTable.around("torch.cpu.Event.synchronize", self.synchronize)
        HookTable.around("torch.cpu.Event.record", self.record)
        HookTable.around("torch.cpu.Stream.wait_event", self.wait_event)
        HookTable.around("torch.cpu.Stream.wait_stream", self.wait_stream)


DisaggregationMode = resolve(symbols().disaggregation_mode)
BatchTokenizedGenerateReqInput = resolve(symbols().batch_tokenized_generate_req_input)


class RequestInput(RoutedCacheKeys):
    def __init__(
        self, peer, vocab_size: int, workload: dict, routes, mode: str, expert_layers: set[int]
    ) -> None:
        self.peer = peer
        self.vocab_size = vocab_size
        self.mode = DisaggregationMode(mode)
        self.arrivals = None
        self.waiting = False
        self.poll_deadline_ns = None
        self.context = None
        self.in_flow = False
        self.submissions = RequestSubmissions(workload)
        self.submissions.content.check_vocabulary(vocab_size)
        self.routing = (
            RoutedContent(self.submissions.content, routes["assignments"], expert_layers)
            if routes is not None
            else None
        )
        if self.routing is not None:
            self.install_cache_keys()

    def recv_pyobj(self, flags):
        if self.arrivals is None:
            advance = self.peer.call("input", waiting=self.waiting, poll_deadline_ns=self.poll_deadline_ns,
                                     context=self.context, in_flow=self.in_flow)
            if advance["kind"] != "advance":
                raise ValueError("Native control expected an advance boundary")
            # One socket receive may read several dispatches. Each keeps its own tokenizer envelope.
            self.arrivals = iter(self.submissions.cohorts(advance["arrivals"]))

        cohort = next(self.arrivals, None)

        # End this native receive poll; the next poll waits for core progress.
        if cohort is None:
            self.arrivals = None
            raise zmq.Again()

        _, indices = cohort
        return BatchTokenizedGenerateReqInput(batch=[self.tokenize(index) for index in indices])

    def tokenize(self, index):
        request = self.submissions.requests[index]
        return self.submissions.tokenized(
            self.submissions.native_id(self.submissions.observed, index),
            self.submissions.content.prompt(request["id"]),
            self.submissions.sampling(index),
            self.vocab_size,
            self.mode,
            extra_key=request["id"] if self.routing is not None else None,
            room=index + 1 if self.mode != DisaggregationMode.NULL else None,
        )

    def request_id(self, identity):
        return self.submissions.requests[self.submissions.index(identity)]["id"]

    def output_token(self, identity: str) -> int:
        return self.submissions.content.output_token(self.request_id(identity))


class NativeRequestReceiver:
    def __init__(self, receiver, scheduler) -> None:
        self.receiver = receiver
        self.scheduler = scheduler
        self.round_started = False
        self.model_ran = False
        self.idle_reported = False
        self.poll_deadline_ns = None

    def poll(self, timeout):
        # SGLang supplies the socket timeout. Simulated time advances at its next receive boundary.
        if self.scheduler.calls.started and timeout >= 0:
            self.poll_deadline_ns = self.scheduler.peer.call("clock") + timeout * 1_000_000

    def recv_requests(self):
        scheduler = self.scheduler

        # The native loop owns this cursor. A PP round includes every slot, not each individual receive.
        first_slot = scheduler.ps.pp_size == 1 or sys._getframe(1).f_locals["mb_id"] == 0
        inbox = scheduler.request_input
        inbox.waiting = False
        inbox.poll_deadline_ns = None
        if first_slot:

            # Native on_idle can precede round-tail work; park only after that work reaches the next receive.
            idle = self.idle_reported and scheduler.is_fully_idle() and not PipelineABI.pending_model_sends(scheduler)
            self.idle_reported = False
            if idle:
                if scheduler.ps.pp_rank == 0 and scheduler.ps.tp_rank == 0:
                    if scheduler.calls.started:

                        # Between arrivals the native loop still polls. The read owns its return and wake conditions.
                        inbox.waiting = True
                        inbox.poll_deadline_ns = self.poll_deadline_ns
                    else:
                        scheduler.peer.call("idle", waiting=True)
                else:
                    scheduler.peer.send({"kind": "idle", "waiting": False})
            self.poll_deadline_ns = None

            # Pending transfers can bypass on_idle; completed rounds must still have a safe external wait.
            inbox.waiting |= (
                self.round_started and not self.model_ran
                and not scheduler.waiting_queue and scheduler.chunked_req is None
                and scheduler.pd is not None and scheduler.pd.waiting()
            )
            self.round_started = True
            self.model_ran = False
        inbox.context = scheduler.calls.scheduler_work()["context"]
        inbox.in_flow = scheduler.calls.started
        return self.receiver.recv_requests()


class PublishedRequests:
    def __init__(self, calls, enabled: bool) -> None:
        self.calls = calls
        self.enabled = enabled

    def send_output(self, output, *args, **kwargs):
        if not self.enabled or self.calls.current.get() is None:
            return
        with self.calls.call("publication", self.send_output) as frame:

            # Prefill's response acknowledges its transfer; only Decode publishes the serving output.
            if self.calls.scheduler.server_args.disaggregation_mode == "prefill":
                return
            frame.outputs = [
                {
                    "request": RequestSubmissions.index(request),
                    "generated_tokens": tokens,
                    "finished": reason is not None,
                    "emits_output": True,
                }
                for request, tokens, reason in zip(
                    output.rids, output.completion_tokens, output.finished_reasons, strict=True
                )
            ]


class FlowExecution(CallSegments):
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


_compute_parallelism_ranks = resolve(symbols().compute_parallelism_ranks)
LogitsProcessorOutput = resolve(symbols().logits_processor_output)
scheduler_module = resolve(symbols().scheduler_module)
Scheduler = resolve(symbols().scheduler)
GenerationBatchResult = resolve(symbols().generation_batch_result)
register_radix_cache_backend = resolve(symbols().register_radix_cache_backend)
model_runner = resolve(symbols().model_runner_module)
PPProxyTensors = resolve(symbols().pp_proxy_tensors)
PortArgs = resolve(symbols().port_args)
ServerArgs = resolve(symbols().server_args)


class NativeScheduler(Scheduler):
    def __init__(self, peer, settings, model_path: str, source, rank: int, workload, workers, local) -> None:
        self.peer = peer
        self.clock_epoch_ns = settings["clock_epoch_ns"]
        self.instance_id = source["id"]
        self.attention_backend = source["attention_backend"]
        worker = source["workers"][rank]
        self.graph_rank = worker["rank"]
        self.group = worker["group"]
        self.launching = False
        self.pd = None
        if source["mode"] != "null":
            from prologue.integrations.frameworks.sglang.control.pd import NativePdBackend

            self.pd = NativePdBackend(self, workers, local)
            self.pd.install_resources()
        self.signals = PipelineSignals(self)
        self.values = TensorValues(f"{source['id']}/{self.graph_rank}")
        self.batch_names = BatchNames()
        self.request_state = RequestStateCalls(observe=settings["observe"])
        model_runner.ModelRunner = partial(
            MetadataRunner, peer=peer, rendezvous=Path(model_path, "group").as_uri(),
            kv_layout=worker["kv_layout"], allocator=source["allocator"],
        )
        cache = source["cache"]
        register_radix_cache_backend("prologue", partial(PrefixCache.build, expected=cache))
        scheduler_module.create_scheduler_watchdog = self.workflow_watchdog
        args = ServerArgs(
            model_path=model_path,
            dtype=source["dtype"].removeprefix("torch."),
            tp_size=source["tp"],
            ep_size=source["ep"],
            pp_size=source["pp"],
            moe_a2a_backend=source["moe_a2a_backend"],
            deepep_mode=source["deepep_mode"],
            pp_max_micro_batch_size=settings["microbatch_requests"],
            device="cpu",
            load_format="dummy",
            skip_tokenizer_init=True,
            sleep_on_idle=True,
            max_total_tokens=settings["max_total_tokens"],
            max_running_requests=settings["max_running_requests"],
            max_prefill_tokens=settings["max_prefill_tokens"],
            page_size=settings["page_size"],
            attention_backend="torch_native",
            disable_radix_cache=cache["disabled"],
            radix_cache_backend="prologue",
            radix_eviction_policy=cache["eviction_policy"],
            chunked_prefill_size=settings["prefill_chunk_tokens"] or -1,
            enable_mixed_chunk=settings["mixed_batches"],
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
            disable_custom_all_reduce=True,
            disaggregation_mode=source["mode"],
            disaggregation_decode_enable_radix_cache=source["mode"] == "decode" and not cache["disabled"],
            random_seed=13,
        )
        if (args.tp_size, args.pp_size, args.ep_size) != (source["tp"], source["pp"], source["ep"]):
            raise ValueError("native configuration changed the supplied parallel topology")
        cp_rank, moe_dp_rank, ep_rank = _compute_parallelism_ranks(args, rank % source["tp"])
        if ep_rank != worker["ep_rank"]:
            raise ValueError("template expert rank differs from native parallel initialization")
        self.experts = None
        has_experts = any(worker["expert_layers"] for worker in source["workers"])
        if has_experts != (settings["expert_routes"] is not None):
            raise ValueError("native model and supplied expert route conditions disagree")
        if has_experts:
            routes = ExpertRoutes(
                settings["expert_routes"], workload["requests"], worker["expert_layers"],
                source["model_config"]["num_hidden_layers"], source["tp"] - 1,
            )
            self.experts = ExpertBatch(routes, source["ep"], ep_rank, args.moe_a2a_backend, args.deepep_mode)
            if source["expert_layout"] != self.experts.layout:
                raise ValueError("native control and source expert layout differ")

        super().__init__(
            args,
            PortArgs.init_new(args),
            gpu_id=0,
            tp_rank=rank % source["tp"],
            moe_ep_rank=ep_rank,
            pp_rank=rank // source["tp"],
            attn_cp_rank=cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=None,
        )
        if settings["mixed_batches"] != self.is_mixed_chunk:
            raise ValueError("native configuration did not retain the requested mixed-batch mode")

        # Preserve native receive, dispatch, selection, result processing and idle housekeeping.
        self.request_input = RequestInput(
            peer,
            self.model_config.vocab_size,
            workload,
            settings["expert_routes"] if not cache["disabled"] else None,
            source["mode"],
            {layer["layer"] for worker in source["workers"] for layer in worker["expert_layers"]},
        )
        self.request_receiver = NativeRequestReceiver(
            replace(self.request_receiver, recv_from_tokenizer=self.request_input), self
        )
        if self.idle_sleeper is not None:
            self.idle_sleeper.poller = self.request_receiver
        self.model_worker.forward_batch_generation = self.model_result
        self.request_state.install()
        self.calls = FlowExecution(self)
        AllocationDemand()
        if self.pd is not None:
            self.pd.install_calls()

        if source["pp"] > 1:
            self.pipeline_messages = PipelineMessages(self.calls, wire_device=platform_domain().device_type)
            self.signals.install()

    def handle_generate_request(self, request):
        # Native initialization binds this method into its dispatcher before call instrumentation is installed.
        self.calls.input(request)
        return super().handle_generate_request(request)

    def description(self, batch):
        if batch is None:
            return {}, self.batch_names.describe(None)
        members = BatchTensors.members(batch)
        for member in members:
            member["id"] = self.request_input.request_id(member["id"])
        work = {
            **self.batch_names.describe(batch),
            "phase": BatchTensors.phase(batch),
            "rank": self.graph_rank,
            "members": members,
        }
        work["context"] = {"members": self.calls.progress(batch.reqs)}
        return BatchTensors.parameters(batch, sum(member["new_tokens"] for member in members)), work

    def action(
        self,
        name,
        parameters,
        work,
        outputs=(),
        *,
        condition=None,
        tensor_inputs=None,
        tensor_outputs=None,
        allocations=None,
    ):
        condition = {
            "phase": work.get("phase", "idle"),
            "state_accesses": RequestStateCalls.condition(work),
            **(condition or {}),
        }

        # Child calls can create batches before their parent resumes; liveness belongs to submission time.
        work["live_batches"] = self.batch_names.live()
        work["live_signals"] = list(self.signals.live)
        if self.pd is not None:
            work["live_signals"].extend(sorted(self.pd.live_signals()))
        work["allocations"] = {name: self.values.allocate(tensor) for name, tensor in (allocations or {}).items()}
        work["values"] = self.values.inputs(tensor_inputs or {})
        work["live_values"] = self.values.live()

        # Native decisions and resource calls have run; execution must complete this return before native continues.
        returned = self.peer.call(
            "action", action=name, condition=condition, parameters=parameters, work=work, outputs=list(outputs)
        )
        self.values.publish(tensor_outputs or {}, returned["values"])
        return returned

    def model_work(self, batch):
        parameters, work = self.description(batch)
        parameters.update(BatchTensors.model_parameters(batch, parameters["input_tokens"]))
        parameters.update(BatchTensors.attention_layout(self.attention_backend, parameters, work["members"]))
        if self.experts is not None:
            work.update(self.experts.bind_batch(parameters, work["members"]))
        return parameters, work

    def run_batch(self, batch, *args, **kwargs):
        self.request_receiver.model_ran = True
        if self.launching:
            return super().run_batch(batch, *args, **kwargs)
        parameters, work = self.model_work(batch)
        with self.request_state.members(batch), self.request_state.record() as accesses:
            result = super().run_batch(batch, *args, **kwargs)
        self.bind_state(work, accesses)

        # The native tensor carries bookkeeping values; later reads must inherit this graph's produced storage.
        outputs = {} if result.next_token_ids is None else {"next_token_ids": result.next_token_ids}
        self.action("run_batch", parameters, work, tensor_outputs=outputs)
        return result

    def _pp_launch_batch(self, mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
        self.request_receiver.model_ran = True
        parameters, work = self.model_work(self.cur_batch)

        # The native launch owns run_batch and its recorded completion; submit the parent graph once.
        self.launching = True
        try:
            with (
                self.signals.producing(),
                self.request_state.members(self.cur_batch),
                self.request_state.record() as accesses,
            ):
                result, event = super()._pp_launch_batch(mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue)
        finally:
            self.launching = False
        work["signals"] = {"completion": event._prologue_signal.identity}
        work["observation_context"] = {"pipeline": event._prologue_signal.context["pipeline"]}
        self.bind_state(work, accesses)
        inputs = (
            {}
            if pp_proxy_tensors is None
            else {f"proxy.{name}": tensor for name, tensor in PipelineABI.proxy_tensors(pp_proxy_tensors).items()}
        )
        if result.pp_hidden_states_proxy_tensors is None:
            outputs = {"next_token_ids": result.next_token_ids}
        else:
            outputs = {
                f"proxy.{name}": tensor
                for name, tensor in PipelineABI.proxy_tensors(result.pp_hidden_states_proxy_tensors).items()
            }
        self.action("launch", parameters, work, tensor_inputs=inputs, tensor_outputs=outputs)
        return result, event

    def _pp_prep_batch_result(self, batch, mb_metadata, pp_outputs):
        parameters, work = self.description(batch)
        work["observation_context"] = {"pipeline": PipelineABI.control_context(self.calls)}
        received = pp_outputs["next_token_ids"]
        tokens = self.next_tokens(batch)

        # Workload token values drive native bookkeeping; the received core value retains its data dependency.
        self.values.alias(received, tokens)
        pp_outputs["next_token_ids"] = tokens
        with self.request_state.members(batch), self.request_state.record() as accesses:
            result = super()._pp_prep_batch_result(batch, mb_metadata, pp_outputs)
        self.bind_state(work, accesses)
        self.action(
            "output_prepare", parameters, work, tensor_inputs={"next_token_ids": tokens},
            tensor_outputs={"next_token_ids": result.next_token_ids},
        )
        return result

    def _pp_make_skip_output_result(self, batch, mb_metadata):
        parameters, work = self.description(batch)
        with self.signals.producing():
            outputs, result, event = super()._pp_make_skip_output_result(batch, mb_metadata)
        work["signals"] = {"completion": event._prologue_signal.identity}
        work["observation_context"] = {"pipeline": event._prologue_signal.context["pipeline"]}
        self.action("output_skip", parameters, work, tensor_outputs={"next_token_ids": result.next_token_ids})
        return outputs, result, event

    def next_tokens(self, batch):
        return torch.tensor([self.request_input.output_token(req.rid) for req in batch.reqs], dtype=torch.int64)

    def bind_state(self, work, accesses):
        work.update({
            name: [self.request_input.request_id(identity) for identity in requests]
            for name, requests in accesses.members.items()
        })

    def model_result(self, batch, **kwargs):
        if not self.pp_group.is_last_rank:
            parameters, work = self.model_work(batch)
            condition = {
                "phase": work["phase"],
                "state_accesses": RequestStateCalls.condition(self.request_state.accesses.get().members),
            }
            outputs = self.peer.call("describe", action="launch", condition=condition, parameters=parameters)
            tensors = {
                name.removeprefix("proxy."): torch.empty(
                    layout["shape"], dtype=getattr(torch, layout["dtype"].removeprefix("torch.")), device="meta",
                )
                for name, layout in outputs.items() if name.startswith("proxy.")
            }
            if not tensors:
                raise ValueError("pipeline launch template has no intermediate tensor outputs")

            # Intermediate names and geometry come from the captured template, not model-family guesses.
            return GenerationBatchResult(pp_hidden_states_proxy_tensors=PPProxyTensors(tensors))

        # These token values drive native request bookkeeping; their device production belongs to the action graph.
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=self.next_tokens(batch),
        )

    @staticmethod
    def workflow_watchdog(*args, **kwargs):
        # Workflow owns wall-clock deadlines; native watchdog diagnostics would race the synchronous resource channel.
        return None

    def process_batch_result(self, batch, result):
        self.calls.process(super().process_batch_result, batch, result)

    def on_idle(self):
        idle = self.is_fully_idle() and not PipelineABI.pending_model_sends(self)
        if idle:
            self.calls.drained()
        super().on_idle()
        self.request_receiver.idle_reported = idle


class BatchNames:
    """Name native batch objects without keeping completed batches alive."""

    def __init__(self) -> None:
        self.names = WeakIdKeyDictionary()
        self.sequence = 0

    def describe(self, batch):
        if batch is not None and batch not in self.names:
            self.names[batch] = self.sequence
            self.sequence += 1
        return {"batch": self.names[batch] if batch is not None else None}

    def live(self):
        return list(self.names.values())
