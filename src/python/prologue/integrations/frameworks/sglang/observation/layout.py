"""Describe model geometry, worker roles and the observed state of native resources."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, update_wrapper
from inspect import signature
from itertools import pairwise
from pathlib import Path
from typing import Any

from prologue.common.io import read_json, write_json
from prologue.integrations.bindings import device_runtime
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.configuration import (
    AttentionCatalog,
    KvBufferAddresses,
    MhaKvLayout,
    ParallelGroups,
)
from prologue.integrations.frameworks.sglang.shared.resources import PoolCoordinates, SourceLayout


class LayerCommunication:
    def __init__(self, native) -> None:
        self.native = native

    @classmethod
    def read(cls, module):
        LayerCommunicator = resolve(symbols().layer_communicator)

        native = getattr(module, "layer_communicator", None)
        return cls(native) if isinstance(native, LayerCommunicator) else None

    def rows(self, mode):
        context = self.native._context

        # Native group widths define token ownership; equal observed dimensions alone do not prove it.
        return ["floor_div", ["mul", "model_tokens", context.process_group_sizes[mode]], context.attn_tp_size]

    def observe(self, owner) -> dict:
        modes = self.native.layer_scatter_modes
        for method, mode in (("prepare_mlp", modes.attn_mode), ("postprocess_layer", modes.mlp_mode)):
            original = getattr(self.native, method)
            hook = partial(self.observe_hidden, owner, self.rows(mode), original)
            setattr(self.native, method, update_wrapper(hook, original))
        return {"input": self.rows(modes.layer_input_mode), "output": self.rows(modes.layer_output_mode)}

    @staticmethod
    def observe_hidden(owner, rows, original, hidden_states, *args, **kwargs):
        if owner.active and owner.selected:
            import torch
            get_attn_tp_context = resolve(symbols().get_attn_tp_context)

            # The ordinary layer ABI owns a flat token axis and a fixed hidden width on this shard.
            with owner.tensors.metadata():
                if isinstance(hidden_states,
                    torch.Tensor) and hidden_states.ndim == 2 and not get_attn_tp_context().input_scattered:
                    owner.tensors.declare_shape(hidden_states, [rows, hidden_states.shape[1]])
        return original(hidden_states, *args, **kwargs)

    def document(self) -> dict:
        native = self.native
        context = native._context
        functions = {}
        for name in (
            "_communicate_simple_fn", "_communicate_with_all_reduce_and_layer_norm_fn",
            "_communicate_summable_tensor_pair_fn", "qkv_latent_func",
        ):
            function = getattr(native, name)
            modes = {}
            if isinstance(function, partial):

                # Native communication factories bind scatter modes that change residual ownership.
                bound = signature(function.func).bind_partial(*function.args, **function.keywords)
                modes = {parameter: mode.name for parameter, mode in bound.arguments.items()}
                function = function.func
            functions[name] = None if function is None else {
                "implementation": f"{function.__module__}.{function.__qualname__}", "bound_modes": modes,
            }

        # Compare initialized policies, not rank placement or the communicator's mutable cache.
        return {
            "implementation": f"{type(native).__module__}.{type(native).__qualname__}",
            "modes": {name: mode.name for name, mode in vars(native.layer_scatter_modes).items()},
            "group_sizes": {mode.name: size for mode, size in context.process_group_sizes.items()},
            "sizes": {name: getattr(context, name)
                for name in ("attn_tp_size", "attn_dp_size", "attn_cp_size", "tp_size")},
            "flags": {name: getattr(native, name) for name in (
                "allow_reduce_scatter", "is_last_layer", "force_layernorm_before_dp_gather",
            )},
            "functions": functions,
        }


class ModuleStructure:
    @classmethod
    def read(cls, module: Any, layers: dict[int, int], storages: dict[int, int]) -> Any:
        import torch
        PPMissingLayer = resolve(symbols().pp_missing_layer)

        if isinstance(module, PPMissingLayer):
            return None
        fields = {}
        identities = {
            "rank",
            "tp_rank",
            "pp_rank",
            "ep_rank",
            "moe_ep_rank",
            "moe_tp_rank",
            "attn_tp_rank",
            "gpu_id",
            "device_id",
            "device_index",
            "local_rank",
            "prefix",
            "layer_name",
        }
        for name, value in sorted(vars(module).items()):
            field = name.lstrip("_")
            if field in identities:
                continue
            if type(value) is int and field in {"layer_id", "layer_idx", "start_layer",
                "end_layer"} and value in layers:

                # A local layer reference is distinct from an unrelated literal with the same numeric value.
                fields[name] = ("local_layer", layers[value])
                continue
            if isinstance(value, torch.device):
                value = value.type
            elif isinstance(value, torch.dtype):
                value = str(value)
            if isinstance(value, (str, int, float, bool, type(None))):
                fields[name] = value
            elif field in {"quant_method", "dispatcher"}:
                fields[name] = f"{type(value).__module__}.{type(value).__qualname__}"

        communicator = LayerCommunication.read(module)
        if communicator is not None:
            fields["layer_communicator"] = communicator.document()

        tensors = []
        for name, value in (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)):
            storage = value.untyped_storage()._cdata
            alias = storages.setdefault(storage, len(storages))
            tensors.append(
                (
                    name,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    str(value.dtype),
                    value.device.type,
                    alias,
                )
            )
        children = []
        for name, child in module.named_children():
            key = cls.read(child, layers, storages)
            if key is not None:
                children.append((str(len(children)) if isinstance(module, torch.nn.ModuleList) else name, key))

        # Placement is bound per worker; native implementation, local geometry and flags remain significant.
        return f"{type(module).__module__}.{type(module).__qualname__}", fields, tensors, children


class NativeWorkerLayout:
    """Read the same initialized worker at a serving or a local acquisition boundary."""

    def __init__(self, runner, revision, rank_offset):
        RadixAttention = resolve(symbols().radix_attention)

        self.model = runner.model
        self.rank_offset = rank_offset
        self.groups = ParallelGroups()
        topology = resolve(symbols().get_parallel)()
        layers = [m.layer_id for m in self.model.modules() if isinstance(m, RadixAttention)]
        self.parallel = {
            "rank": rank_offset + runner.pp_rank * runner.tp_size + runner.tp_rank,
            "tp_rank": runner.tp_rank, "pp_rank": runner.pp_rank,
            "logical_tp": runner.tp_size, "logical_pp": runner.pp_size,
            "logical_ep": topology.moe_ep_size, "ep_rank": topology.moe_ep_rank,
            "layer_range": [min(layers), max(layers) + 1],
        }
        args = runner.server_args
        self.implementation = {
            "framework_revision": revision,
            "model": runner.model_config.hf_config.to_dict(),
            "dtype": str(runner.model_config.dtype),
            "attention_backend": args.attention_backend,
            "page_size": args.page_size,
            "load_format": args.load_format,
            "disaggregation_mode": args.disaggregation_mode,
        }

    def pools(self, pools, server_args):
        allocator = pools.token_to_kv_pool_allocator
        allocator_type, cache_type = type(allocator), type(pools.tree_cache)
        self.implementation.update(
            allocator=f"{allocator_type.__module__}.{allocator_type.__qualname__}",
            cache={"type": f"{cache_type.__module__}.{cache_type.__qualname__}",
                   "disabled": pools.disable_radix_cache, "eviction_policy": server_args.radix_eviction_policy.lower()},
        )
        self.kv_layout = MhaKvLayout.from_pool(allocator.get_kvcache())

    def roles(self, sites, expert_layers, conditions):
        facts = {
            "parallel": self.parallel, "groups": self.groups.document(self.rank_offset),
            "kv_layout": self.kv_layout.document(), "expert_layers": expert_layers,
            "modules": [path for path, _ in self.model.named_modules() if path in sites],
        }
        return WorkerRolePlan(self.model, facts, conditions)

    def environment(self, scheduler, roles, runtime):
        self.implementation["scheduling"] = {
            "chunked_prefill": scheduler.chunked_prefill_size is not None, "mixed_batches": scheduler.is_mixed_chunk,
        }
        role = roles[str(self.parallel["rank"])]
        return {
            "framework": "sglang", **runtime, "parallel": self.parallel, "implementation": self.implementation,
            "kv_layout": self.kv_layout.document(), "worker_roles": roles,
            **{key: role[key] for key in ("expert_layers", "kv_peers") if key in role},
        }


class WorkerRolePlan:
    def __init__(self, model: Any, facts: dict[str, Any], observation_conditions) -> None:
        import torch.distributed as distributed
        get_world_group = resolve(symbols().get_world_group)

        parallel = facts["parallel"]
        identity = parallel["rank"]
        layout = dict(facts["kv_layout"])
        first, last = layout["start_layer"], layout["end_layer"]
        layers = {layer: index for index, layer in enumerate(range(first, last + 1))}
        layout.update(start_layer=0, end_layer=last - first)
        facts = {
            **facts,
            "observation_conditions": observation_conditions,
            "key": (
                parallel["tp_rank"] == 0,
                parallel["pp_rank"] == 0,
                parallel["pp_rank"] + 1 == parallel["logical_pp"],
                [
                    (name, group["role"], len(group["members"]), group["members"][0] == identity)
                    for name, group in facts["groups"].items()
                ],
                layout,
                ModuleStructure.read(model, layers, {}),
            ),
        }
        group = get_world_group().cpu_group
        self.workers = [None] * distributed.get_world_size(group)

        # PP forwards control requests after local handling, so whole-world exchange belongs to initialization.
        distributed.all_gather_object(self.workers, facts, group=group)

    @staticmethod
    def observed_ranks(workers, bindings):
        selected = {worker["representative"] for worker in bindings.values()}
        for representative in sorted(selected):
            group = {worker["parallel"]["rank"]: set(worker["observation_conditions"]) for worker in workers
                     if bindings[str(worker["parallel"]["rank"])]["representative"] == representative}
            missing = set().union(*group.values()) - group[representative]
            while missing:
                rank = min(group, key=lambda rank: (-len(group[rank] & missing), rank))
                selected.add(rank)
                missing.difference_update(group[rank])
        return sorted(selected)

    def document(self, peers) -> dict[str, Any]:
        representatives = []
        bindings = {}
        for initialized in self.workers:
            worker = dict(initialized)
            if peers is not None:
                rank = worker["parallel"]["rank"]
                worker["kv_peers"] = peers.layouts(rank)
                worker["key"] = (worker["key"], peers.signature(rank))
            source = next((previous for previous in representatives if previous["key"] == worker["key"]), None)
            if source is None:
                source = worker
                representatives.append(worker)
            bindings[str(worker["parallel"]["rank"])] = {
                "representative": source["parallel"]["rank"],
                **{name: value for name, value in worker.items() if name not in {"key", "observation_conditions"}},
            }

        return {
            "basis": "native_module_layout_group_roles_and_source_conditions",
            "participants": len(self.workers),
            "representatives": [worker["parallel"]["rank"] for worker in representatives],
            "observed_roles": self.observed_ranks(self.workers, bindings),
            "workers": bindings,
        }


class NativeKvFacts:
    def __init__(self, directory: Path, rank: int) -> None:
        self.directory = directory
        self.recording = json.loads((directory / f"rank-{rank}.json").read_text())
        self.registrations = {}

    def registration(self, recording):
        observed = recording["registration"]
        layout = MhaKvLayout.read(recording["layout"])
        buffers = KvBufferAddresses(
            layout, recording["page_size"], observed["pointers"], observed["lengths"], observed["page_bytes"],
        )
        return recording["parallel"], buffers, observed["metadata"]

    def metadata_accesses(self, scope, parent, remote):
        transfer = scope["native_transfer"]
        chunk = parent["native_kv_chunk"]
        member, = parent["work"]["members"]
        source = self.registrations[self.recording["kv_session"]][2]
        target = self.registrations[transfer["session"]][2]
        if (transfer["room"], transfer["source_index"]) != (chunk["room"], chunk["metadata_index"]):
            raise ValueError("native metadata send differs from its enqueued request")
        if not source["row_bytes"] or source["row_bytes"] != target["row_bytes"]:
            raise ValueError("native metadata endpoints have different row layouts")
        if parent["parameters"]["last_chunk"] != 1:
            raise ValueError("native first-token metadata must belong to the final chunk")

        # Native send_aux transfers one complete registered row from every buffer, including the room gate.
        rows = []
        for buffers, index in ((source, transfer["source_index"]), (target, transfer["target_index"])):
            if buffers["reservations"][str(transfer["room"])] != {"request": member["id"], "index": index}:
                raise ValueError("native metadata transfer differs from its endpoint's request reservation")
            addresses = []
            for pointer, length, width in zip(
                buffers["pointers"], buffers["lengths"], buffers["row_bytes"], strict=True
            ):
                if width <= 0 or length % width or not 0 <= index < length // width:
                    raise ValueError("native metadata row lies outside its registered buffer")
                addresses.append(pointer + index * width)
            rows.append(addresses)
        expected = [list(block) for block in zip(*rows, source["row_bytes"], strict=True)]
        if sorted(transfer["blocks"]) != sorted(expected):
            raise ValueError("native metadata transfer does not cover the registered request rows")

        accesses = []
        for pool, mode in (("local", "read"), (remote, "write")):
            key = f"metadata/{mode}"
            accesses.append({
                "space": "slot", "pool": pool, "field": "metadata", "access": mode,
                "members": key, "requests": [member["id"]], "interval": "service",
            })

        # Native senders independently publish the same request's first token; the receiver joins their completions.
        accesses[-1]["publication"] = "first-token"
        return accesses

    @staticmethod
    def logical_regions(buffers, address, length, pages, member):
        page_size = buffers.page_size
        begin = member["context_tokens"]
        end = begin + member["new_tokens"]
        regions = []
        for region in buffers.regions(address, length):
            token = region["token_begin"]
            while token < region["token_end"]:
                page, offset = divmod(token, page_size)
                count = min(region["token_end"] - token, page_size - offset)
                logical = begin + pages[page] * page_size + offset
                stop = min(logical + count, end)
                if logical < stop:
                    regions.append((region["layer"], region["slice"], logical, stop))
                token += count
        return regions

    def accesses(self, scope):
        transfer = scope["native_transfer"]
        parent = self.recording["scopes"][scope["submitted_by"]]
        target_rank, target, _ = self.registrations[transfer["session"]]
        remote = self.destinations[target_rank["rank"]]
        if transfer["kind"] == "metadata":
            return self.metadata_accesses(scope, parent, remote)
        member, = parent["work"]["members"]
        chunk = parent["native_kv_chunk"]
        if transfer["source_pages"] != chunk["source_pages"] or len(transfer["source_pages"]) != len(
            transfer["target_pages"]
        ):
            raise ValueError("native transport changed the enqueued chunk's page extent")
        if transfer["source_pages"] and not transfer["blocks"]:
            raise ValueError("nonempty native KV transfer has no observed address blocks")
        source = self.registrations[self.recording["kv_session"]][1]
        if source.page_size != target.page_size:
            raise ValueError("native transfer endpoints have different page sizes")
        positions = {}
        for endpoint in ("source", "target"):
            pages = transfer[f"{endpoint}_pages"]
            positions[endpoint] = {int(page): index for index, page in enumerate(pages)}
            if len(positions[endpoint]) != len(pages):
                raise ValueError("native transfer repeats a physical page in one request chunk")

        intervals = defaultdict(list)
        for source_address, target_address, length in transfer["blocks"]:
            for pool, access, buffers, address, pages in (
                ("local", "read", source, source_address, positions["source"]),
                (remote, "write", target, target_address, positions["target"]),
            ):
                for layer, byte_slice, begin, end in self.logical_regions(buffers, address, length, pages, member):
                    key = pool, access, layer, byte_slice["token_bytes"], byte_slice["begin"], byte_slice["end"]
                    intervals[key].append((begin, end))
        result = []
        for (pool, access, layer, width, first, last), ranges in sorted(intervals.items()):
            merged = []
            for begin, end in sorted(ranges):
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([begin, end])
            expected = [member["context_tokens"], member["context_tokens"] + member["new_tokens"]]
            if merged != [expected]:
                raise ValueError("native transfer slice does not cover its declared request chunk")
            value = {
                "space": "kv", "pool": pool, "layer": layer, "access": access, "interval": "service",
                "slice": {"token_bytes": width, "begin": first, "end": last},
                "regions": [{"request": member["id"], "begin": expected[0], "end": expected[1]}],
                "rule": {"begin": "context_tokens", "end": ["add", "context_tokens", "new_tokens"]},
            }

            # Adjacent K/V pieces with the same logical coverage form one token payload, not separate templates.
            adjacent = result and all(result[-1][key] == value[key] for key in ("pool", "layer", "access", "regions"))
            if adjacent and result[-1]["slice"]["token_bytes"] == width and result[-1]["slice"]["end"] == first:
                result[-1]["slice"]["end"] = last
            else:
                result.append(value)
        return result

    def connect_metadata(self, facts):
        raw = self.recording["scopes"]
        transfers = [scope for scope in raw if scope.get("native_transfer", {}).get("kind") == "metadata"]
        if not transfers:
            return

        writers = defaultdict(list)
        for index, scope in enumerate(facts["scopes"]):
            for access in scope["slot_accesses"]:
                if (access["pool"], access["field"], access["access"]) == ("local", "metadata", "write"):
                    for request in access["requests"]:
                        writers[request].append((index, access["finish"]))

        for transfer in transfers:
            reader, = [access for access in facts["scopes"][transfer["id"]]["slot_accesses"]
                       if access["pool"] == "local" and access["access"] == "read"]
            request, = reader["requests"]
            (writer, completed), = writers[request]
            caller = transfer["submitted_by"]
            while caller is not None and caller != raw[writer]["parent"]:
                caller = raw[caller]["parent"]
            if caller is None:
                raise ValueError("native first-token row was not produced by its enqueue invocation")

            # This registered row consumes the invocation's data; the edge declares dataflow, not a runtime fence.
            facts["dependencies"].append({"source": completed, "target": reader["start"], "kind": "state"})

    def document(self):
        transfers = [scope for scope in self.recording["scopes"] if "native_transfer" in scope]
        if not transfers:
            return self.recording

        # The target registration is independent native evidence; source pointer arithmetic cannot invent its layout.
        pools = self.directory / self.recording.get("kv_pool_directory", "kv-pools")
        for path in sorted(pools.glob("*.json")):
            recording = json.loads(path.read_text())
            session = recording["registration"]["session"]
            if session in self.registrations:
                raise ValueError("native KV session identifies more than one pool")
            self.registrations[session] = self.registration(recording)

        # Canonical peer positions come from actual transfer endpoints, not a second run of native routing code.
        targets = {scope["native_transfer"]["session"] for scope in transfers}
        ordered = sorted(
            (self.registrations[session][0] for session in targets),
            key=lambda parallel: (parallel["pp_rank"], parallel["tp_rank"]),
        )
        self.destinations = {parallel["rank"]: f"destination/{index}" for index, parallel in enumerate(ordered)}
        if set(self.destinations.values()) != set(self.recording["environment"]["kv_peers"]):
            raise ValueError("observed native transfers do not cover the declared peer interfaces")
        for scope in transfers:
            transfer = scope["native_transfer"]
            rank, buffers, _ = self.registrations[transfer["session"]]
            binding = self.destinations[rank["rank"]]
            expected = {"layout": buffers.layout.document(), "tp": rank["logical_tp"], "pp": rank["logical_pp"]}
            if expected != self.recording["environment"]["kv_peers"][binding]:
                raise ValueError("native transfer registration differs from the planned peer interface")
            scope["state_accesses"] = self.accesses(scope)

        # KV and metadata have different inputs; retain each service and expose only the chunk's final completion.
        chunks = defaultdict(list)
        for scope in transfers:
            chunks[scope["submitted_by"]].append(scope)
        for parent, services in chunks.items():
            services.sort(key=lambda scope: scope["host_interval"]["begin_counter"])
            if (any(scope["parent"] != parent or scope["thread"] != services[0]["thread"] for scope in services)
                    or any(a["host_interval"]["end_counter"] > b["host_interval"]["begin_counter"]
                           for a, b in pairwise(services))):
                raise ValueError("native chunk services lack one serial synchronous caller")
            self.recording["scopes"][parent]["ports"].append({
                "name": "transfer.complete", "direction": "output", "use": "signal",
                "boundary": {"scope": services[-1]["id"], "interval": "service", "side": "end"},
            })
        return self.recording

    def protocol_timings(self):
        recording = self.recording
        if (
            recording["environment"]["implementation"]["disaggregation_mode"] == "null"
            or recording["kv_session"] is None
        ):
            return None

        reservations, transfers = [], []
        for scope in recording["scopes"]:
            if "reservation_publication" in scope:
                reservations.append(scope["reservation_publication"])
            if "native_transfer" in scope:
                parent = scope["submitted_by"]
                member, = recording["scopes"][parent]["work"]["members"]
                transfer = scope["native_transfer"]
                transfers.append({
                    "request": member["id"], "chunk": parent, "kind": transfer["kind"],
                    "destination": transfer["session"], "end_raw_ns": scope["host_interval"]["end_raw_ns"],
                })
        return {
            "session": recording["kv_session"], "endpoint": recording["kv_endpoint"],
            "states": recording["protocol_states"], "reservations": reservations, "transfers": transfers,
            "receipts": recording["reservation_receipts"],
        }


def observed_graph(directory, rank, recording, trace, *, evidence):
    runtime = device_runtime()
    recording["scopes"] = SourceLayout(recording["environment"], runtime.tensor_layouts).lower(recording["scopes"])
    return runtime.observation(directory, rank, recording, trace, evidence=evidence)


def serving_facts(directory, rank, trace, *, evidence):
    native = NativeKvFacts(directory, rank)
    recording = native.document()
    observation = observed_graph(directory, rank, recording, trace, evidence=evidence)
    observation.name_streams(*PipelineABI.stream_declarations(recording))
    facts = observation.document()
    native.connect_metadata(facts)
    if evidence and (timing := native.protocol_timings()) is not None:
        facts["measurements"]["pd_protocol"] = timing
    return facts


@dataclass
class IndexValues:
    tensor: object
    offset: int

    def document(self):
        return [PoolCoordinates.logical(int(value), self.offset) for value in self.tensor.tolist()]


class NativeResourceHistory:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.events = []
        self.reads = []
        self.bindings = []
        self.initial = None
        self.final = None
        self.fields = {}
        self.state_events = []
        self.state_free = {}

    @property
    def enabled(self):
        return self.observation.active and self.observation.selected

    def indices(self, values, offset):
        if values is None:
            return None

        # Freeze mutable device views asynchronously; CPU materialization happens only after the window drains.
        with self.observation.tensors.metadata():
            return IndexValues(values.detach().clone(), offset)

    @staticmethod
    def node(value):
        return None if value is None else value.id

    def available(self):
        # Diagnostic snapshots call the original readers, never the hooks for real control observations.
        return {name: int(original()) // unit for name, (original, unit) in self.capacity_readers.items()}

    def begin(self):
        self.initial = {
            "page_size": self.page_size,
            "page_capacity": self.allocator.size // self.page_size,
            "free_slots": [PoolCoordinates.logical(int(slot)) for slot in self.slots.free_slots],
            "free_pages": self.indices(self.allocator.free_pages, 1),
            "released_pages": self.indices(self.allocator.release_pages, 1),
            "available": self.available(),
            "state_slots": {
                field: {"capacity": pool.size, "free": list(pool.free_slots)} for field, pool in self.fields.items()
            },
        }

    def wrap(self, owner, method, hook, *bound):
        original = getattr(owner, method)

        def observed(*args, **kwargs):
            if self.enabled:
                return hook(*bound, original, *args, **kwargs)
            return original(*args, **kwargs)

        setattr(owner, method, update_wrapper(observed, original))
        return original

    def bind_field(self, field, allocator):
        self.fields[field] = allocator
        operation = {"kind": "state_slots", "field": field, "operation": "available"}
        self.wrap(allocator, "available_size", self.read_capacity, operation, 1)
        for name, operation in (("alloc", "allocate"), ("free", "release")):
            self.wrap(allocator, name, self.state_operation, field, operation)

    def state_operation(self, field, operation, original, *args):
        record = {"field": field, "operation": operation}
        if operation == "release":
            record["slot"], = args
        self.state_events.append(record)
        record["begin_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result = original(*args)
        record["end_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if operation == "allocate":
            record["slot"] = result
        return result

    def state_binding(self, field, request, slot):
        if self.enabled:
            self.state_events.append(
                {
                    "field": field,
                    "operation": "bind",
                    "request": self.observation.content.aliases[request.rid],
                    "slot": slot,
                }
            )

    def batch(self, scope, batch):
        if not self.enabled:
            return

        # Read the table consumed by attention, including cached prefixes, rather than replaying allocator results.
        with self.observation.tensors.metadata():
            for req, slot, length in zip(
                batch.reqs, batch.req_pool_indices_cpu.tolist(), batch.seq_lens_cpu.tolist(), strict=True
            ):
                self.bindings.append({
                    "scope": scope, "request": self.observation.content.aliases[req.rid],
                    "slot": PoolCoordinates.logical(int(slot)), "begin": 0,
                    "indices": self.indices(self.slots.req_to_token[slot, :length], self.page_size),
                })

    @staticmethod
    def page_ranges(indices, page_size, capacity):
        ranges = []
        for index in indices:
            if not 0 <= index < capacity:
                raise ValueError("native request binding points outside the KV pool")
            page, offset = divmod(index, page_size)
            if ranges and ranges[-1][0] == page and ranges[-1][2] == offset:
                ranges[-1][2] += 1
            else:
                ranges.append([page, offset, offset + 1])
        return ranges

    @contextmanager
    def event(self, operation, *, query=False):
        capture = self.observation.capture
        scope = capture.current.get()

        # Resource wrappers can suspend the current segment; its invocation still encloses the whole operation.
        if capture.scopes[scope]["role"] == "segment":
            scope = capture.scopes[scope]["parent"]
        record = {
            "scope": scope, "thread": threading.get_native_id(),
            "operation": operation, "begin_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        }
        (self.reads if query else self.events).append(record)
        yield record
        record["end_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if not query:
            record["available"] = self.available()

    def read_capacity(self, operation, unit, original, *args, **kwargs):
        with self.event(operation, query=True) as record:
            record["control"] = not self.observation.calls.resources.in_resource.get()
            result = original(*args, **kwargs)
        record["result"] = int(result) // unit
        return result

    def allocate_slots(self, original, reqs):
        requests = [self.observation.content.aliases[req.rid] for req in reqs]
        with self.event({"kind": "slots", "requests": requests}) as record:
            result = original(reqs)
        record["result"] = None if result is None else [PoolCoordinates.logical(int(slot)) for slot in result]
        return result

    def release_slot(self, original, req):
        operation = {
            "kind": "release_request", "request": self.observation.content.aliases[req.rid],
            "slot": PoolCoordinates.logical(int(req.req_pool_idx)),
        }
        with self.event(operation):
            return original(req)

    def allocate_pages(self, original, *args, **kwargs):
        with self.event({"kind": "allocate", "entry": original.__name__}) as record:
            result = original(*args, **kwargs)
        record["indices"] = self.indices(result, self.page_size)
        return result

    def release_pages(self, original, free_index):
        operation = {
            "kind": "release_pages", "grouped": not self.allocator.is_not_in_free_group,
            "indices": self.indices(free_index, self.page_size),
        }
        with self.event(operation):
            return original(free_index)

    def match(self, original, params):
        operation = {"kind": "prefix", "operation": "match", "key_tokens": list(params.key.raw_token_ids())}
        with self.event(operation) as record:
            result = original(params)
        record["result"] = {
            "node": self.node(result.last_device_node),
            "indices": self.indices(result.device_indices, self.page_size),
        }
        return result

    def insert(self, original, params):
        operation = {
            "kind": "prefix", "operation": "insert", "key_tokens": list(params.key.raw_token_ids()),
            "indices": self.indices(params.value, self.page_size),
        }
        with self.event(operation) as record:
            result = original(params)
        record["result"] = {"node": self.node(result.last_device_node), "existing_tokens": result.prefix_len}
        return result

    def lock(self, acquire, original, node, *args, **kwargs):
        with self.event({"kind": "prefix", "operation": "lock", "node": self.node(node), "acquire": acquire}) as record:
            result = original(node, *args, **kwargs)
        record["result"] = result.delta
        return result

    def evict(self, original, params):
        with self.event({"kind": "prefix", "operation": "evict", "tokens": params.num_tokens}) as record:
            result = original(params)
        record["result"] = result.num_tokens_evicted
        return result

    def bind(self, cache, allocator, slots):
        self.cache, self.allocator, self.slots = cache, allocator, slots
        self.page_size = allocator.page_size
        self.capacity_readers = {}
        for owner, method, resource, unit in (
            (slots, "available_size", "slots", 1),
            (allocator, "available_size", "pages", self.page_size),
            (cache, "evictable_size", "evictable_tokens", 1),
            (cache, "protected_size", "protected_tokens", 1),
        ):
            operation = {"kind": "capacity", "resource": resource}
            self.capacity_readers[resource] = self.wrap(owner, method, self.read_capacity, operation, unit), unit

        for owner, methods in (
            (slots, {"alloc": self.allocate_slots, "free": self.release_slot}),
            (allocator, {"alloc": self.allocate_pages, "alloc_extend": self.allocate_pages,
                         "alloc_decode": self.allocate_pages, "free": self.release_pages}),
            (cache, {"match_prefix": self.match, "insert": self.insert, "inc_lock_ref": partial(self.lock, True),
                     "dec_lock_ref": partial(self.lock, False), "evict": self.evict}),
        ):
            for name, hook in methods.items():
                self.wrap(owner, name, hook)

    @classmethod
    def materialize(cls, value):
        if isinstance(value, IndexValues):
            return value.document()
        if isinstance(value, dict):
            return {key: cls.materialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.materialize(item) for item in value]
        return value

    def close(self):
        # Native flush resets the pools next; preserve the end-of-serving state before it runs.
        self.final = self.available()
        self.state_free = {field: list(pool.free_slots) for field, pool in self.fields.items()}
        self.initial = self.materialize(self.initial)
        self.events = self.materialize(self.events)
        self.bindings = self.materialize(self.bindings)
        for binding in self.bindings:
            binding["ranges"] = self.page_ranges(binding.pop("indices"), self.page_size, self.allocator.size)

    def document(self):
        return {
            "rank": self.observation.parallel["rank"], "clock": "monotonic_raw_ns",
            "initial": self.initial, "final": self.final, "events": self.events, "bindings": self.bindings,
            "state_events": self.state_events,
            "state_free": self.state_free,
            "reads": self.reads,
        }


def publish_cuts(source, work, rank, identities, runner, output):
    catalog = AttentionCatalog.extract(
        source, work, rank, identities, read_json(runner), device_runtime().scoped_tasks.source
    )
    write_json(output, catalog)
