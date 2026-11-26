"""Observe live scheduler, model and CPU timing at the native serving boundaries."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from prologue.common.io import write_json
from prologue.integrations.bindings import route
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import symbols
from prologue.integrations.frameworks.sglang.observation.call_ports import NativeModelObservation
from prologue.integrations.frameworks.sglang.observation.communications import (
    NativeCommunications,
)
from prologue.integrations.frameworks.sglang.observation.experts import ExpertScopes, NativeContent, NativeExpertRoutes
from prologue.integrations.frameworks.sglang.observation.host_timing import received_requests
from prologue.integrations.frameworks.sglang.observation.kv_scopes import NativeKvScopes
from prologue.integrations.frameworks.sglang.observation.model import ModelScopes
from prologue.integrations.frameworks.sglang.observation.pipeline import PipelineScopes
from prologue.integrations.frameworks.sglang.observation.resource_history import NativeResourceHistory
from prologue.integrations.frameworks.sglang.observation.serving import FlowScopes
from prologue.integrations.frameworks.sglang.observation.window import NativeProfiles, ObservationWindow
from prologue.integrations.frameworks.sglang.observation.worker_layout import NativeWorkerLayout
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration
from prologue.integrations.frameworks.sglang.shared.metadata_calls import MetadataTransfers
from prologue.integrations.frameworks.sglang.shared.pipeline_calls import PipelineABI


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
