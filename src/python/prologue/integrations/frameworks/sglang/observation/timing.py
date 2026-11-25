"""Use native worker idle and reset boundaries for lightweight serving measurements."""

from contextlib import nullcontext
from pathlib import Path
from time import CLOCK_MONOTONIC_RAW, clock_gettime_ns

from prologue.common.io import write_json
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.content import NativeContent
from prologue.integrations.frameworks.sglang.observation.host_samples import NativeHostSamples
from prologue.integrations.frameworks.sglang.observation.module_scopes import ModelScopes
from prologue.integrations.frameworks.sglang.observation.native_routes import NativeExpertRoutes
from prologue.integrations.frameworks.sglang.observation.window import ObservationWindow
from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI


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
        if self.host is not None:
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
        BatchTokenizedGenerateReqInput = resolve(symbols().batch_tokenized_generate_req_input)
        TokenizedGenerateReqInput = resolve(symbols().tokenized_generate_req_input)

        messages = original(receiver)
        if not messages or receiver.ps.pp_rank != 0:
            return messages
        received_ns = clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        for message in messages:
            if isinstance(message, BatchTokenizedGenerateReqInput):
                requests = message
            elif isinstance(message, TokenizedGenerateReqInput):
                requests = (message,)
            else:
                continue
            for request in requests:
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
            if self.host is not None:
                record = self.host.document()
            elif self.attention is not None:
                record = self.attention.document()
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
