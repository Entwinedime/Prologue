"""Observe live scheduler, model and CPU timing at the native serving boundaries."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial, update_wrapper
from inspect import signature
from pathlib import Path
from threading import get_native_id
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns
from typing import Any

from prologue.common.io import write_json
from prologue.integrations.bindings import route
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.communications import (
    DeviceCollectiveCuts,
    HostPipelineScopes,
    NativeCommunications,
    PipelineScopes,
)
from prologue.integrations.frameworks.sglang.observation.layout import NativeResourceHistory, NativeWorkerLayout
from prologue.integrations.frameworks.sglang.observation.model import (
    ExpertScopes,
    ModelScopes,
    NativeContent,
    NativeExpertRoutes,
    NativeModelObservation,
)
from prologue.integrations.frameworks.sglang.observation.serving import FlowScopes, NativeKvScopes, ServingCuts
from prologue.integrations.frameworks.sglang.observation.window import NativeProfiles, ObservationWindow
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestStateCalls
from prologue.integrations.frameworks.sglang.shared.calls import MetadataTransfers, PipelineABI, native_implementation
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration, ParallelGroups
from prologue.integrations.runtimes.pytorch.capture import CpuProtocol


class NativeHostSamples(ServingCuts):
    def __init__(self, aliases, description, host_profile):
        super().__init__(aliases, description["mode"])
        self.rank_offset = description["rank_offset"]
        self.ep_size = description["ep_size"]
        self.enabled = False
        self.request_state = RequestStateCalls(observe=False)
        self.parent = ContextVar("timed_host_scope", default=0)
        self.scopes = [
            {"id": 0, "parent": None, "role": "window", "site": "", "implementation": "", "work": {}, "bindings": {}}
        ]
        self.auxiliary_intervals = []
        self.host_profile = host_profile
        self.profile = None
        self.communications = DeviceCollectiveCuts(self)
        self.protocol = None
        self.pipeline = None
        if description["pp_size"] > 1:
            self.protocol = CpuProtocol(
                description["native_participants"], self.communicating, self.parent, self.protocol_service,
                transport_timing=True,
            )
            self.pipeline = HostPipelineScopes(self)

    def start(self, scheduler, rank, directory):
        self.scheduler = scheduler
        self.rank = rank
        self.groups = ParallelGroups()
        self.collective_groups = self.groups.document(self.rank_offset)
        self.profile = self.host_profile(directory)
        self.profile.start()
        if self.protocol is not None:
            self.protocol.begin()

    @staticmethod
    @contextmanager
    def protocol_service(interval):
        interval["begin_raw_ns"] = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        try:
            yield
        finally:
            interval["end_raw_ns"] = clock_gettime_ns(CLOCK_MONOTONIC_RAW)

    @contextmanager
    def scope(
        self, role, site, implementation, *, work=None, condition=None, parameters=None, bindings=None, dimensions=None
    ):
        occurrence = {
            "id": len(self.scopes), "parent": self.parent.get(), "role": role, "site": site,
            "implementation": implementation, "work": {} if work is None else dict(work),
            "bindings": {} if bindings is None else dict(bindings), "thread": get_native_id(),
        }
        if condition is not None:
            occurrence["bindings"]["condition"] = condition
        if parameters is not None:
            occurrence["parameters"] = parameters
        if dimensions is not None:
            occurrence["dimensions"] = dimensions
        self.scopes.append(occurrence)
        previous = self.parent.set(occurrence["id"])
        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        try:
            yield occurrence
        finally:
            occurrence["raw_interval"] = [begin, clock_gettime_ns(CLOCK_MONOTONIC_RAW)]
            self.parent.reset(previous)

    def native_call(self, role, original, *, implementation=None, **facts):
        implementation = implementation or native_implementation(original)
        return self.scope(role, original.__qualname__, implementation, **facts)

    @property
    def collective_at_root(self):
        return self.parent.get() == 0

    @contextmanager
    def local_exchange(self, operation, source, _outputs):
        owner = self.parent.get()
        if self.enabled and owner:
            stamp = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            with self.auxiliary():
                self.scopes[owner].setdefault("local_submissions", []).append({
                    "operation": operation, "dtype": str(source.dtype), "bytes": source.numel() * source.element_size(),
                    "thread": get_native_id(), "submitted_raw_ns": stamp,
                })
        yield

    def observe_exchange(self, original, role, group, root_rank, operand, run):
        work, parameters, bindings = None, None, None
        if operand is not None and operand.output is None:
            parameters = PipelineABI.tensor_parameters(operand.tensor)
        if role == "gather":
            group_role = self.groups.name(group)
            bindings = {"collective": PipelineABI.gather_condition(operand.tensor, operand.gather_dim, group_role)}
            if self.pipeline is not None:
                work = self.pipeline.messages.work()
        with self.native_call(role, original, work=work, parameters=parameters, bindings=bindings):
            return run()

    def observe_expert_exchange(self, original, dispatcher, *args, **kwargs):
        if not self.enabled:
            return original(dispatcher, *args, **kwargs)
        with self.native_call("communication", original):
            return original(dispatcher, *args, **kwargs)

    def collective(self, original, boundary, arguments):
        role = boundary.condition["collective"]["group"]
        with self.native_call(boundary.operation, original, work=boundary.work, condition=boundary.condition) as scope:
            scope["group"] = self.collective_groups[role]
            scope["parameters"] = boundary.parameters
            return original(**arguments)

    def enclosing(self, frame):
        return self.native_call("invocation", frame.original, work=frame.work)

    @contextmanager
    def fragment(self, frame):
        with self.native_call("segment", frame.original) as occurrence:
            yield
        occurrence["bindings"]["condition"] = frame.observed_condition
        occurrence["work"]["continuation"] = frame.continuation

    @contextmanager
    def boundary(self, original, boundary):
        with self.native_call(boundary.operation, original, work=boundary.work,
                              implementation=boundary.implementation) as occurrence:
            yield
        occurrence["work"]["context"] = boundary.work["context"]
        occurrence["parameters"] = boundary.parameters

    @contextmanager
    def read_boundary(self, kind, original, work, condition):
        with self.native_call("query", original, work=work, condition=condition):
            yield {}

    def load_model(self, original, runner):
        result = original(runner)
        self.attach_model(runner.model)
        return result

    def attach_model(self, model):
        modules = dict(model.named_modules())
        for site, role in ModelScopes.selected(modules).items():
            self.attach(site, role, modules[site])

    def attach(self, site, role, module, method="forward"):
        original = getattr(module, method)
        identity = type(module) if method == "forward" else original
        implementation = f"{identity.__module__}.{identity.__qualname__}"
        body = signature(original) if role == "model_body" else None
        wrapped = partial(self.module, role, site, implementation, body, original)
        setattr(module, method, update_wrapper(wrapped, original))

    def observe_expert_layers(self, experts):
        for site, module in experts.items():
            self.attach(site, "experts", module)
            self.attach(module.run_moe_core.__qualname__, "expert_compute", module, "run_moe_core")

    def module(self, role, site, implementation, body_signature, original, *args, **kwargs):
        if not self.enabled:
            return original(*args, **kwargs)
        with self.scope(role, site, implementation) as scope:
            if body_signature is not None:

                # Read the tensors passed to the model after the native backend has applied padding.
                with self.auxiliary():
                    arguments = ModelScopes.bound_arguments(body_signature, args, kwargs)
                    scope["dimensions"] = ModelScopes.model_dimensions(arguments)
            return original(*args, **kwargs)

    def run_batch(self, original, scheduler, batch, *args, **kwargs):
        work = self.batch_work(batch)
        with self.auxiliary():
            members = work["members"]
            dimensions = BatchTensors.input_dimensions(sum(member["new_tokens"] for member in members), len(members))
        with self.native_call("run_batch", original, work=work, dimensions=dimensions):
            return original(scheduler, batch, *args, **kwargs)

    def forward(self, original, runner, batch, *args, **kwargs):
        with self.auxiliary():
            dimensions = BatchTensors.input_dimensions(batch.input_ids.numel(), batch.batch_size)
        with self.native_call("model", original, dimensions=dimensions):
            return original(runner, batch, *args, **kwargs)

    def sample(self, original, runner, *args, **kwargs):
        with self.native_call("sample", original):
            return original(runner, *args, **kwargs)

    @contextmanager
    def auxiliary(self):
        if not self.enabled:
            yield
            return
        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        try:
            yield
        finally:
            self.auxiliary_intervals.append([get_native_id(), begin, clock_gettime_ns(CLOCK_MONOTONIC_RAW)])

    def install(self, device_api):
        if self.protocol is not None:
            MetadataTransfers(self, self.protocol).install()
        HookTable.around(symbols().model_runner_load_model, self.load_model)
        ModelScopes.install_batch_hooks(self)
        self.communications.install(self.observe_expert_exchange if self.ep_size > 1 else None)
        if self.mode != "null":
            NativeKvScopes(self).install()
        super().install()
        if self.pipeline is not None:
            self.pipeline.install(device_api)

    def document(self):
        document = {
            "clock": "monotonic_raw_ns", "scopes": self.scopes,
            "auxiliary_intervals": self.auxiliary_intervals, "timing": self.profile.finish(),
        }
        if self.protocol is not None:
            document["protocol"] = self.protocol.document()
        return document


def received_requests(messages):
    batch_type = resolve(symbols().batch_tokenized_generate_req_input)
    request_type = resolve(symbols().tokenized_generate_req_input)
    for message in messages or ():
        if isinstance(message, batch_type):
            yield from message
        elif isinstance(message, request_type):
            yield message


class NativeTiming:
    def __init__(self, description, *, host_profile=None, specializations=None, attention=None):
        self.directory = Path(description["instance_directory"]) / "timing"
        self.rank_offset = description["rank_offset"]
        self.pipeline_stages = description["pp_size"]
        self.window = ObservationWindow(Path(description["output"]), description)
        self.content = NativeContent(description)
        self.host = None
        self.attention = attention(self.window, description) if attention else None
        self.routing = None
        if host_profile is not None:
            self.host = NativeHostSamples(self.content.aliases, description, host_profile)
            self.content.auxiliary = self.host.auxiliary
        elif self.attention is None:
            self.specializations = specializations
            self.requests = {native: {"id": identity} for native, identity in self.content.aliases.items()}
        if "ep" in description["serving_input"]["configuration"]:
            auxiliary = nullcontext if self.host is None else self.host.auxiliary
            self.routing = NativeExpertRoutes(
                description, self.content.execution_aliases, metadata=auxiliary, auxiliary=auxiliary
            )

    def load_routed_model(self, original, runner):
        result = original(runner)
        experts, layers = ModelScopes.expert_structure(dict(runner.model.named_modules()))
        self.routing.configure(experts, layers, runner.model_config.hf_config.num_hidden_layers)
        if self.host is not None:
            self.host.observe_expert_layers(experts)
        if self.content.route_assignments is not None:
            workers = [{"expert_layers": layers}]
            if self.pipeline_stages > 1:
                import torch.distributed as distributed
                get_world_group = resolve(symbols().get_world_group)

                # Every stage loads its model before serving; cache keys need all expert layer identities.
                group = get_world_group().cpu_group
                workers = [None] * distributed.get_world_size(group)
                distributed.all_gather_object(workers, {"expert_layers": layers}, group=group)
            self.content.configure(workers)
        return result

    def rank(self, scheduler):
        coordinates = scheduler.ps
        return self.rank_offset + coordinates.pp_rank * coordinates.tp_size + coordinates.tp_rank

    def request(self, original, scheduler, request):
        self.window.begin(request.rid)
        if self.host is not None and request.rid in self.window.requests["source"]:
            self.host.enabled = True
        return original(scheduler, request)

    def receive(self, original, receiver):
        messages = original(receiver)
        if not messages or receiver.ps.pp_rank != 0:
            return messages
        received_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        for request in received_requests(messages):
            if request.rid in self.requests:
                self.requests[request.rid].setdefault("ingress_raw_ns", received_ns)
        return messages

    def publish(self, original, sender, output, *args, **kwargs):
        BatchTokenIDOutput = resolve(symbols().batch_token_id_output)

        result = original(sender, output, *args, **kwargs)
        if sender.socket is None or not isinstance(output, BatchTokenIDOutput) or not output.rids:
            return result

        # Only the actual sending worker has a socket. Record before any serialization of measurements.
        published_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        for rid, count, finished in zip(output.rids, output.completion_tokens, output.finished_reasons, strict=True):
            if rid not in self.requests or count <= 0:
                continue
            row = self.requests[rid]
            if "output_tokens" in row:
                raise ValueError(f"invalid native token publication for {rid}")
            row.setdefault("first_publication_raw_ns", published_ns)
            if finished is not None:
                row.update(last_publication_raw_ns=published_ns, output_tokens=count)
        return result

    def flush(self, original, scheduler, empty_cache=True):
        result = original(scheduler, empty_cache=False)
        if not result or not self.window.drained:
            return result
        rank = self.rank(scheduler)
        directory = self.directory / f"rank-{rank}"
        if self.window.phase == "warmup":
            directory.mkdir(parents=True)
            if self.host is not None:
                self.host.start(scheduler, rank, directory)
            elif self.attention is None:
                self.specializations.start()
        elif self.window.phase == "source":
            probe = self.host or self.attention
            if probe is not None:
                record = probe.document()
            else:
                requests = [row for row in self.requests.values() if len(row) > 1]
                record = {"requests": requests, "specialization_loads": self.specializations.finish()}
            write_json(directory / "record.json", {**record, "rank": rank}, replace=False)

        self.window.prepared_phase = self.window.phase
        return result

    def idle(self, original, scheduler):
        if scheduler.is_fully_idle() and not PipelineABI.pending_model_sends(scheduler):
            rank = self.rank(scheduler)
            self.window.group_idle(rank)
            if self.window.inputs_complete() and not self.window.drained:
                if self.host is not None:
                    self.host.enabled = False
                self.window.drain(rank)
            if self.window.prepared_phase is not None:
                self.window.complete(rank, scheduler.idle_sleeper is not None)
        return original(scheduler)

    def install(self, device_api):
        if self.host is not None:
            self.host.install(device_api)
        elif self.attention is not None:
            self.attention.install()
        else:
            HookTable.around(symbols().pull_raw_reqs, self.receive)
            HookTable.around(symbols().sender_wrapper_send_output, self.publish)
        self.content.install()
        if self.routing is not None:
            HookTable.around(symbols().model_runner_load_model, self.load_routed_model)
            self.routing.install()
        HookTable.around(symbols().scheduler_handle_generate_request, self.request)
        HookTable.around(symbols().scheduler_flush_cache, self.flush)
        HookTable.around(symbols().scheduler_on_idle, self.idle)


class SGLangObservation(NativeModelObservation):
    def __init__(self, description: dict[str, Any], markers) -> None:

        self.description = description
        self.framework_revision = NativeConfiguration.upstream_revision(route().revision)
        self.directory = Path(description["output"])
        self.kv_pools = self.directory / "kv-pools"
        self.window = ObservationWindow(self.directory, description)
        super().__init__(markers, description["mode"], resource_trace=description["resource_trace"])
        self.content = NativeContent(description, self.tensors.auxiliary)
        self.aliases = self.content.aliases
        if description["resource_trace"]:

            self.resource_history = NativeResourceHistory(self)
        self.parallel = {}
        self.routing = None
        if "ep" in description["serving_input"]["configuration"]:
            self.routing = NativeExpertRoutes(
                description,
                self.content.execution_aliases,
                metadata=self.tensors.metadata,
                auxiliary=self.tensors.auxiliary,
            )

    @property
    def protocol_enabled(self):
        # Nonrepresentative workers still supply protocol evidence until the source group drains.
        return self.window.phase == "source" and not self.window.drained

    def load_model(self, original: Any, runner: Any) -> Any:

        result = original(runner)
        self.content.content.check_vocabulary(runner.model_config.vocab_size)
        self.worker = NativeWorkerLayout(runner, self.framework_revision, self.description["rank_offset"])
        self.parallel, self.groups = self.worker.parallel, self.worker.groups
        self.implementation = self.worker.implementation
        self.tensors.namespace = f"rank/{self.parallel['rank']}/"
        self.modules = ModelScopes(self)
        self.modules.attach_model(runner.model)
        if self.routing is not None:

            self.routing.configure(
                self.modules.experts, self.modules.expert_layers, runner.model_config.hf_config.num_hidden_layers
            )
            self.experts = ExpertScopes(self, self.modules.experts)
        return result

    def prepare_roles(self) -> None:
        # All model stages are initialized together; a later scheduler request cannot run a world collective.
        sites = set(self.modules.bindings)
        if self.routing is not None:
            sites.update(self.experts.sites)
        conditions = (
            () if self.routing is None else self.routing.geometry.observation_conditions(self.routing.input["requests"])
        )
        self.roles = self.worker.roles(sites, self.modules.expert_layers, conditions)
        if any(worker["expert_layers"] for worker in self.roles.workers) != (self.routing is not None):
            raise ValueError(
                "MoE models require a declared expert topology and routes; dense models do not accept them"
            )
        self.content.configure(self.roles.workers)

    def plan_roles(self):
        peers = None
        if self.description["mode"] != "null":
            from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import PdPeerBindings

            workers = {}
            for path in self.kv_pools.glob("*.json"):
                worker = json.loads(path.read_text())
                workers[worker["parallel"]["rank"]] = worker
            peers = PdPeerBindings(workers)

        # The shared warmup barrier already published every endpoint; final selection needs no native collective.
        plan = self.roles.document(peers)
        self.worker_roles = plan["workers"]
        if self.description["observe_all_workers"]:
            plan["observed_roles"] = self.description["native_participants"]
        if self.parallel["pp_rank"] == 0 and self.parallel["tp_rank"] == 0:
            write_json(Path(self.description["instance_directory"]) / "role-plan.json", plan, indent=2)
        return plan["observed_roles"]

    def request(self, original: Any, scheduler: Any, request: Any) -> Any:
        self.window.begin(request.rid)
        if self.window.phase == "source" and self.selected and not self.active:
            activity = None
            if self.description["service_reference"]:
                activity = self.directory / f"rank-{self.parallel['rank']}-thread-activity.jsonl"
            self.window_scope = self.capture.runtime.window(activity)
            self.window_scope.__enter__()
            self.active = True
            if self.resource_history is not None:
                self.resource_history.begin()
        return original(scheduler, request)

    def receive(self, original: Any, receiver: Any, *, ingress=False) -> Any:
        attempt = None
        if (ingress and self.resource_history is not None and self.selected
                and receiver.ps.pp_rank == 0 and receiver.ps.attn_tp_rank == 0 and receiver.ps.attn_cp_rank == 0):

            # The first source pull precedes profiler activation; warmup requests have no source identities.
            context = self.calls.scheduler_work()["context"] if self.window.phase in (None, "source") else None
            attempt = {"context": context, "begin_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)}

        requests = original(receiver)
        received_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if ingress and receiver.ps.pp_rank != 0:
            return requests
        identities = [request.rid for request in received_requests(requests)]
        if attempt is not None:
            attempt["message_count"] = len(requests or ())
        self.window.receive(identities, received_ns, ingress=ingress, attempt=attempt)
        return requests

    def can_flush(self) -> bool:
        if not self.scheduler.is_fully_idle() or PipelineABI.pending_model_sends(self.scheduler):
            return False

        # Input dispatch may be mid-pipeline round; only on_idle closes the observed control flow.
        return self.window.phase is None or self.window.drained

    def flush_wrapper(
        self, original: Any, wrapper: Any, *, flush_cache: Any, is_fully_idle: Any, ipc_channels: Any
    ) -> None:
        self.scheduler = flush_cache.__self__
        original(wrapper, flush_cache=flush_cache, is_fully_idle=self.can_flush, ipc_channels=ipc_channels)

    def drain(self, scheduler: Any) -> None:
        if PipelineABI.pending_model_sends(scheduler):
            return
        if not scheduler.is_fully_idle() or self.capture.current.get() != 0:
            return
        self.window.group_idle(self.parallel["rank"])
        if not self.window.inputs_complete() or self.window.drained:
            return

        # Drain the observed device tails before native flush changes logical KV state.
        self.capture.runtime.synchronize()
        if self.active:
            self.window_scope.__exit__(None, None, None)
            self.active = False
            if self.resource_history is not None:
                self.resource_history.close()
        self.window.drain(self.parallel["rank"])

    def flush(self, original: Any, scheduler: Any, empty_cache: bool = True) -> Any:
        result = original(scheduler, empty_cache=False)
        if result and self.window.drained:
            registration = self.kv_transfers.registration()
            if registration is not None:

                # Every endpoint supplies layout evidence even when its computation is represented by another worker.
                write_json(self.kv_pools / f"{self.parallel['rank']}.json", {
                    "mode": self.description["mode"], "parallel": self.parallel, "layout": self.kv_layout.document(),
                    "page_size": self.implementation["page_size"], "registration": registration,
                })
            if self.window.phase == "source" and self.selected:
                self.active = False
                self.communications.bind_returned_operands()
                recording = self.document({
                    **self.worker.environment(scheduler, self.worker_roles, self.capture.runtime.environment()),
                    "initial_state": {"logical_kv": "empty_after_native_flush", "workspace": "uniform_native_warmup"},
                    "closure": {"native_idle": True, "device_drained": True, "pending_native_transfers": 0},
                }, self.calls.queries)
                recording["request_receipts"] = self.window.receipts
                recording["request_receive_rounds"] = self.window.receive_rounds
                recording["request_ingress"] = self.window.ingress
                recording["protocol_states"] = self.kv_transfers.states
                recording["reservation_receipts"] = self.kv_transfers.receipts
                recording["kv_session"] = None if registration is None else registration["session"]
                recording["kv_endpoint"] = None if registration is None else registration["control_endpoint"]
                if registration is not None and self.description["local_shards"]:
                    recording["kv_pool_directory"] = str(self.kv_pools.relative_to(self.directory))
                if self.resource_history is not None:
                    recording["request_input_attempts"] = self.window.input_attempts
                    recording["action_values"] = self.action_values
                    write_json(
                        self.directory / "resources" / f"rank-{self.parallel['rank']}.json",
                        self.resource_history.document(),
                    )
                write_json(self.directory / f"rank-{self.parallel['rank']}.json", recording)
            self.window.prepared_phase = self.window.phase
        return result

    def idle(self, original: Any, scheduler: Any) -> Any:
        if scheduler.is_fully_idle():
            self.drain(scheduler)
            if self.window.prepared_phase is not None:
                self.window.complete(self.parallel["rank"], scheduler.idle_sleeper is not None)
        return original(scheduler)

    def install(self, device_api, shard_transport) -> None:

        receiver = symbols().scheduler_request_receiver
        HookTable.group({
            symbols().model_runner_load_model: self.load_model,
            symbols().scheduler_handle_generate_request: self.request,
            f"{receiver}.recv_requests": self.receive,
            f"{receiver}._pull_raw_reqs": partial(self.receive, ingress=True),
            symbols().scheduler_flush_cache: self.flush,
            symbols().scheduler_on_idle: self.idle,
            symbols().scheduler_flush_wrapper_init: self.flush_wrapper,
        }).install()
        ModelScopes.install_batch_hooks(self)
        self.request_state.install()
        self.calls = FlowScopes(self)
        self.communications = NativeCommunications(self)
        if shard_transport is not None:

            # CPU protocol observers must see the original device operand, before staging changes its device.
            shard_transport(self.directory, self.tensors.auxiliary, self.communications.local_exchange).install(
                self.description
            )
        profiles = NativeProfiles(self)
        profiles.install()

        # Keep metadata timing inside the device dispatch hooks and outside the caller's template work.
        MetadataTransfers(self.calls, profiles.protocol).install()
        self.communications.install()
        self.kv_transfers = NativeKvScopes(self.calls, self)
        self.kv_transfers.install()
        self.calls.install()
        if self.description["pp_size"] > 1:

            self.pipeline = PipelineScopes(self)
            self.pipeline.install(device_api)
        self.content.install()
        if self.routing is not None:
            self.routing.install()
