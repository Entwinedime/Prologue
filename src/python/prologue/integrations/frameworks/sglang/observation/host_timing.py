"""Observe host-only services, requests and completed token publications."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial, update_wrapper
from inspect import signature
from pathlib import Path
from threading import get_native_id
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.common.io import write_json
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.communications import (
    DeviceCollectiveCuts,
)
from prologue.integrations.frameworks.sglang.observation.experts import NativeContent, NativeExpertRoutes
from prologue.integrations.frameworks.sglang.observation.kv_scopes import NativeKvScopes
from prologue.integrations.frameworks.sglang.observation.model import ModelScopes
from prologue.integrations.frameworks.sglang.observation.pipeline import HostPipelineScopes
from prologue.integrations.frameworks.sglang.observation.probes.batch_timing import BatchTiming
from prologue.integrations.frameworks.sglang.observation.serving import ServingCuts
from prologue.integrations.frameworks.sglang.observation.window import ObservationWindow
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors, RequestStateCalls
from prologue.integrations.frameworks.sglang.shared.calls import native_implementation
from prologue.integrations.frameworks.sglang.shared.configuration import ParallelGroups
from prologue.integrations.frameworks.sglang.shared.metadata_calls import MetadataTransfers
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI
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
        self.communications = DeviceCollectiveCuts(self, self.record_auxiliary)
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
        self, role, site, implementation, *, work=None, condition=None, parameters=None, bindings=None,
        dimensions=None, entered=None,
    ):
        observed_begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW) if entered is None else entered
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
            end = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
            occurrence.setdefault("raw_interval", [begin, end])
            self.parent.reset(previous)
            self.record_auxiliary(observed_begin, occurrence)

    def record_auxiliary(self, entered, occurrence=None):
        if occurrence is None:
            self.auxiliary_intervals.append([get_native_id(), entered, clock_gettime_ns(CLOCK_MONOTONIC_RAW)])
            return
        native_begin, native_end = occurrence["raw_interval"]
        thread = occurrence["thread"]
        self.auxiliary_intervals.append([thread, entered, native_begin])
        self.auxiliary_intervals.append([thread, native_end, clock_gettime_ns(CLOCK_MONOTONIC_RAW)])

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

    def observe_exchange(self, original, role, group, root_rank, operand, run, *, entered=None):
        work, parameters, bindings = None, None, None
        if operand is not None and operand.output is None:
            parameters = PipelineABI.tensor_parameters(operand.tensor)
        if role == "gather":
            group_role = self.groups.name(group)
            bindings = {"collective": PipelineABI.gather_condition(operand.tensor, operand.gather_dim, group_role)}
            if self.pipeline is not None:
                work = self.pipeline.messages.work()

        with self.native_call(
            role, original, work=work, parameters=parameters, bindings=bindings, entered=entered
        ) as scope:
            return self.timed_call(scope, run)

    def observe_expert_exchange(self, original, dispatcher, *args, **kwargs):
        if not self.enabled:
            return original(dispatcher, *args, **kwargs)
        with self.native_call("communication", original) as scope:
            return self.timed_call(scope, original, dispatcher, *args, **kwargs)

    def collective(self, original, boundary, arguments):
        role = boundary.condition["collective"]["group"]
        with self.native_call(boundary.operation, original, work=boundary.work, condition=boundary.condition) as scope:
            scope["group"] = self.collective_groups[role]
            scope["parameters"] = boundary.parameters
            return self.timed_call(scope, original, **arguments)

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
        with self.native_call("query", original, work=work, condition=condition) as occurrence:
            yield {"native_read": original, "scope": occurrence}

    def native_read(self, query, original, *args, **kwargs):
        if query is None or query["native_read"] is not original:
            return original(*args, **kwargs)

        # Only the outer native read owns the service interval, including any internal clock calls.
        occurrence = query["scope"]
        occurrence["direct_native_call"] = True
        return self.timed_call(occurrence, original, *args, **kwargs)

    @staticmethod
    def timed_call(occurrence, original, *args, **kwargs):
        begin = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        result = original(*args, **kwargs)
        occurrence["raw_interval"] = [begin, clock_gettime_ns(CLOCK_MONOTONIC_RAW)]
        return result

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

        enter_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        dimensions = None
        if body_signature is not None:

            # Read the tensors passed to the model after the native backend has applied padding.
            arguments = ModelScopes.bound_arguments(body_signature, args, kwargs)
            dimensions = ModelScopes.model_dimensions(arguments)
        with self.scope(role, site, implementation, dimensions=dimensions, entered=enter_ns) as scope:
            return self.timed_call(scope, original, *args, **kwargs)

    def run_batch(self, original, scheduler, batch, *args, **kwargs):
        enter_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        work = self.batch_work(batch)
        members = work["members"]
        dimensions = BatchTensors.input_dimensions(sum(member["new_tokens"] for member in members), len(members))
        with self.native_call("run_batch", original, work=work, dimensions=dimensions, entered=enter_ns) as scope:
            return self.timed_call(scope, original, scheduler, batch, *args, **kwargs)

    def forward(self, original, runner, batch, *args, **kwargs):
        enter_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        dimensions = BatchTensors.input_dimensions(batch.input_ids.numel(), batch.batch_size)
        with self.native_call("model", original, dimensions=dimensions, entered=enter_ns) as scope:
            return self.timed_call(scope, original, runner, batch, *args, **kwargs)

    def sample(self, original, runner, *args, **kwargs):
        enter_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        with self.native_call("sample", original, entered=enter_ns) as scope:
            return self.timed_call(scope, original, runner, *args, **kwargs)

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
        self.batch_timing = BatchTiming(self.window, self.content.aliases) if description["batch_timing"] else None
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
                if self.batch_timing is not None:
                    record["batch_timing"] = self.batch_timing.document()

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
        if self.batch_timing is not None:
            self.batch_timing.install()
        if self.routing is not None:
            HookTable.around(symbols().model_runner_load_model, self.load_routed_model)
            self.routing.install()

        HookTable.around(symbols().scheduler_handle_generate_request, self.request)
        HookTable.around(symbols().scheduler_flush_cache, self.flush)
        HookTable.around(symbols().scheduler_on_idle, self.idle)
