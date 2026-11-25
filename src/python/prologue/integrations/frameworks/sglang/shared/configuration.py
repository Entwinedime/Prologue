"""Model, serving and worker layout shared by native capture and prediction."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import accumulate

from prologue.common.cost import WorkCatalog
from prologue.common.workload import ExpertRoutes
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols


class NativeConfiguration:

    @staticmethod
    def upstream_revision(value: str | None) -> str:
        if not isinstance(value, str) or re.fullmatch("[0-9a-f]{40}", value) is None:
            raise ValueError("SGLang image requires its actual upstream Git revision metadata")
        return value

    def __init__(self, source: dict) -> None:
        allowed = {
            "id", "tp", "pp", "ep", "attention_backend", "page_size", "max_running_requests", "max_total_tokens",
            "max_prefill_tokens", "prefill_chunk_tokens", "mixed_batches", "prefix_cache", "microbatch_requests",
        }
        if source.keys() - allowed:
            raise ValueError("native configuration contains unsupported fields")
        self.values = {"pp": 1, "prefill_chunk_tokens": 0, "mixed_batches": False, "prefix_cache": False, **source}
        positive = ["tp", "pp", "page_size", "max_running_requests", "max_total_tokens", "max_prefill_tokens"]
        if "microbatch_requests" in self.values:
            positive.append("microbatch_requests")
        for name in positive:
            value = self.values[name]
            if type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(f"native {name} must be a positive signed 64-bit integer")
        if self.values["max_total_tokens"] < self.values["page_size"]:
            raise ValueError("native token capacity must contain at least one KV page")

        chunk = self.values["prefill_chunk_tokens"]
        if type(chunk) is not int or not 0 <= chunk < 1 << 63:
            raise ValueError("native prefill_chunk_tokens must be a nonnegative signed 64-bit integer")
        for name in ("mixed_batches", "prefix_cache"):
            if type(self.values[name]) is not bool:
                raise TypeError(f"native {name} must be a boolean")
        if self.values["mixed_batches"] and not chunk:
            raise ValueError("mixed batches require chunked prefill")
        if not isinstance(self.values["attention_backend"], str) or not self.values["attention_backend"]:
            raise ValueError("native attention_backend must be a nonempty string")

        self.tp, self.pp = self.values["tp"], self.values["pp"]
        self.ep = 1
        if "ep" in self.values:
            expert = self.values["ep"]
            if expert.keys() != {"id", "size"} or not isinstance(expert["id"], str) or not expert["id"]:
                raise ValueError("an EP topology needs a nonempty id and size")
            self.ep = expert["size"]
            if type(self.ep) is not int or not 0 < self.ep < 1 << 63:
                raise ValueError("native EP size must be a positive signed 64-bit integer")

    @staticmethod
    def batch_capacity(configuration: dict) -> int:
        slots = configuration["max_running_requests"]

        # Scheduler initializes the default per PP stage; admission also checks the shared request pool.
        microbatch = configuration.get("microbatch_requests", max(slots // configuration.get("pp", 1), 1))
        return min(slots, microbatch)

    def routes(self, workload: dict):
        if "ep" not in self.values:
            return None
        return workload["ep_routes"][self.values["ep"]["id"]]

    def model_token_alignment(self) -> int:
        # This adapter enables eager MLP synchronization through DeepEP.
        return self.tp if self.ep > 1 else 1

    def structure(self) -> dict:
        # Capacities and chunk length are target parameters; changing the native mechanism needs another source.
        return {
            "tp": self.tp, "pp": self.pp, "ep": self.ep,
            "attention_backend": self.values["attention_backend"], "page_size": self.values["page_size"],
            "prefix_cache": self.values["prefix_cache"],
            "chunked_prefill": self.values["prefill_chunk_tokens"] > 0,
            "mixed_batches": self.values["mixed_batches"],
        }

    def settings(self) -> dict:
        return {name: self.values[name] for name in (
            "max_running_requests", "max_total_tokens", "max_prefill_tokens", "page_size", "prefill_chunk_tokens",
            "microbatch_requests",
        ) if name in self.values}

    def runner_options(self, model):
        values = self.values
        options = {
            "model_path": model["path"],
            "dtype": model["dtype"],
            "device": platform_domain().device_type,
            "tp_size": self.tp,
            "pp_size": self.pp,
            "attention_backend": values["attention_backend"],
            **{
                key: values[key]
                for key in ("page_size", "max_running_requests", "max_total_tokens", "max_prefill_tokens")
            },
            "chunked_prefill_size": values["prefill_chunk_tokens"] or -1,
            "enable_mixed_chunk": values["mixed_batches"],
            "disable_radix_cache": not values["prefix_cache"],
            "disable_overlap_schedule": True,
            "disable_cuda_graph": True,
            "skip_tokenizer_init": True,
            "load_format": "dummy",
            "sleep_on_idle": True,
            "trust_remote_code": True,
            "mem_fraction_static": 0.8,
            "log_level": "warning",
        }
        if "microbatch_requests" in values:
            options["pp_max_micro_batch_size"] = values["microbatch_requests"]
        if self.ep > 1:
            options.update(ep_size=self.ep, moe_a2a_backend="deepep", deepep_mode="normal")
        return options

    def control(self) -> dict:
        return {
            "mixed_batches": self.values["mixed_batches"], "disable_radix_cache": not self.values["prefix_cache"],
            "moe_a2a_backend": "deepep" if self.ep > 1 else "none", "deepep_mode": "normal" if self.ep > 1 else "auto",
        }


@dataclass
class InstanceConfiguration:
    identity: str
    mode: str
    configuration: NativeConfiguration
    ranks: range


class ServingInput:
    """Normalize model, workload and instances; each entrypoint owns its execution options."""

    def __init__(self, source: dict) -> None:
        self.model = source["model"]
        if self.model.keys() != {"path", "dtype"}:
            raise ValueError("serving model needs path and dtype")
        if self.model["dtype"] not in ("float16", "bfloat16", "float32"):
            raise ValueError("serving model dtype must be explicit float16, bfloat16 or float32")

        self.instances = []
        self.worker_count = 0
        identities = set()
        for description in source["instances"]:
            if description.keys() != {"id", "mode", "configuration"}:
                raise ValueError("a serving instance needs id, mode and configuration")
            identity = description["id"]
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("serving instance ids must be unique nonempty strings")
            identities.add(identity)
            configuration = NativeConfiguration(description["configuration"])

            # Each PP stage owns one TP group. EP partitions that group on the same workers.
            ranks = range(self.worker_count, self.worker_count + configuration.pp * configuration.tp)
            self.instances.append(InstanceConfiguration(identity, description["mode"], configuration, ranks))
            self.worker_count = ranks.stop

        modes = [instance.mode for instance in self.instances]
        if modes != ["null"] and sorted(modes) != ["decode", "prefill"]:
            raise ValueError("serving requires one ordinary instance or one prefill/decode pair")
        if len({instance.configuration.values["page_size"] for instance in self.instances}) != 1:
            raise ValueError("native PD transfer requires equal page sizes")
        self.workload = self.workload_input(source["workload"])
        for instance in self.instances:
            routes = instance.configuration.routes(self.workload)
            if routes is not None:

                # Validate declared positions before loading a model; native geometry later checks their consumers.
                ExpertRoutes.padding_input(routes)

    @staticmethod
    def workload_input(source: dict) -> dict:
        workload = dict(source)
        if workload.keys() - {"id", "requests", "prefixes", "ep_routes"}:
            raise ValueError("serving workload contains unsupported fields")
        workload.setdefault("prefixes", [])
        requests = workload["requests"]
        if not isinstance(requests, list) or not requests:
            raise ValueError("serving needs a nonempty request array")
        identities = set()
        for request in requests:
            if request.keys() - {"id", "arrival_ns", "prompt_tokens", "output_tokens", "prefix_id"}:
                raise ValueError("serving request contains unsupported fields")
            identity = request["id"]
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("request IDs must be unique nonempty strings")
            identities.add(identity)
            for name in ("arrival_ns", "prompt_tokens", "output_tokens"):
                minimum = 0 if name == "arrival_ns" else 1
                if type(request[name]) is not int or not minimum <= request[name] < 1 << 63:
                    raise ValueError(f"invalid signed 64-bit request {name}: {identity}")
            if request["prompt_tokens"] + request["output_tokens"] - 1 >= 1 << 63:
                raise ValueError(f"request token length overflows: {identity}")
        return workload


class CaptureInput(ServingInput):
    def __init__(self, source: dict) -> None:
        required = {"model", "instances", "workload", "observe_all_workers"}
        if not required <= source.keys() or source.keys() - required - {"submission_groups", "warmup"}:
            raise ValueError("a native job needs one model, explicit instances, workload and observation role policy")
        if type(source["observe_all_workers"]) is not bool:
            raise TypeError("observation role policy must be a boolean")
        super().__init__(source)
        self.observe_all_workers = source["observe_all_workers"]
        if "warmup" in source and self.workload_input(source["warmup"]) != self.workload:
            raise ValueError("native warmup must use the observed workload and submission groups")
        identities = [request["id"] for request in self.workload["requests"]]
        self.submission_groups = source.get("submission_groups", [identities])
        if not isinstance(self.submission_groups, list) or not self.submission_groups or any(
            not isinstance(group, list) or not group or any(not isinstance(identity, str) for identity in group)
            for group in self.submission_groups
        ):
            raise ValueError("capture submission groups must be nonempty request-ID arrays")
        members = [identity for group in self.submission_groups for identity in group]
        if len(members) != len(identities) or set(members) != set(identities):
            raise ValueError("capture submission groups must partition the workload requests")


class ParallelGroups:
    def __init__(self):
        import torch.distributed as distributed
        get_parallel = resolve(symbols().get_parallel)

        parallel = get_parallel()
        self.groups = {"default": distributed.group.WORLD}
        for name in ("world", "tp", "pp", "attn_tp", "attn_cp", "moe_dp", "moe_ep", "moe_tp"):
            group = getattr(parallel, name + "_group")
            self.groups[name] = group.device_group
            self.groups[name + "_cpu"] = group.cpu_group

    def name(self, group):
        # Native aliases share the same process-group object, not merely the same participants.
        for name, candidate in self.groups.items():
            if candidate is group:
                return name
        raise ValueError("native communication uses a group outside the declared parallel topology")

    def is_cpu(self, group):
        # Device-free control also stores device metadata on CPU; native group roles retain the distinction.
        return any(candidate is group for name, candidate in self.groups.items() if name.endswith("_cpu"))

    def document(self, rank_offset):
        import torch.distributed as distributed

        return {
            name: {
                "role": self.name(group),
                "channel": f"collective/{group.group_name}",
                "members": [rank_offset + rank for rank in distributed.get_process_group_ranks(group)],
            }
            for name, group in self.groups.items()
        }


class WorkerRoles:
    role_fields = ("parallel", "kv_layout", "expert_layers", "kv_peers")

    @classmethod
    def bind_worker(cls, domain, role):
        return domain | {name: role[name] for name in cls.role_fields if name in role}

    @staticmethod
    def rank(domain):
        matches = [role["parallel"]["rank"] for role in domain["worker_roles"].values()
                   if all(role["parallel"].get(key) == value for key, value in domain["parallel"].items())]
        if len(matches) != 1:
            raise ValueError("work coordinates require one observed logical worker")
        return matches[0]


class AttentionCatalog:
    @staticmethod
    def runner(job):
        instances = job["instances"]
        if len(instances) == 1 and instances[0]["mode"] == "null":
            settings = instances[0]["configuration"]
            if settings["tp"] == settings.get("pp", 1) == 1 and "ep" not in settings:
                return {"model": job["model"], "configuration": settings}
        return None

    @classmethod
    def extract(cls, source, work, rank, identities, runner, task_reader):
        catalog = WorkCatalog(work / f"{rank}-graph.json", identities)
        facts = json.loads((work / f"{rank}-facts.json").read_bytes())
        return cls.from_catalog(source, work / f"{rank}-sdk/trace.json", rank, catalog, facts, runner, task_reader)

    @staticmethod
    def from_catalog(source, trace, rank, catalog, facts, runner, task_reader):
        scopes = {}
        for piece, definition, _, features in catalog:
            scope = piece["scope"]
            if (
                piece["kind"] == "device"
                and features["phase"] == "decode"
                and catalog.definitions[definition]["role"] == "attention"
                and not catalog.definitions[definition]["children"]
            ):
                scopes[scope] = definition
        observed = task_reader(source, trace, rank, catalog, scopes, facts) if scopes else {}

        # A shared definition can bind without a layer selector only when its source instances agree on ownership.
        groups = defaultdict(list)
        for scope, definition in scopes.items():
            groups[definition].append(scope)
        cuts = []
        for index, instances in groups.items():
            first = instances[0]
            parts = observed.get(first)
            if parts and all(observed.get(scope) == parts for scope in instances):
                cuts.append({"definition": catalog.definitions[index], "site": catalog.graph["scopes"][first]["site"],
                             "parts": parts, "source_calls": len(instances)})
        return {"domain": catalog.domain, "runner": runner, "cuts": cuts}


class ExpertBatch:
    def __init__(self, routes: ExpertRoutes, size: int, rank: int, backend: str, mode: str) -> None:
        self.routes = routes
        self.size = size
        self.rank = rank
        if size == 1 and backend == "none":
            self.layout = f"{platform_domain().name}_standard"
            self.compute_rows = "expert_input_tokens"
        elif size > 1 and (backend, mode) == ("deepep", "normal"):
            self.layout = f"{platform_domain().name}_deepep_normal_alltoall"
            self.compute_rows = "expert_received_tokens"
        else:
            raise ValueError("unsupported native expert boundary layout")

        # Parameters address the local expert-layer sequence; workload routes retain actual model layer IDs.
        self.layer_parameters = {layer: f"expert/{index}/received" for index, layer in enumerate(routes.layers)}
        if any(layer["experts"] % size for layer in routes.layers.values()):
            raise ValueError("EP group must partition the native routed experts")

    def observation_conditions(self, requests):
        prompts = {request["id"]: request["prompt_tokens"] for request in requests}
        conditions = set()
        for local, (layer, geometry) in enumerate(sorted(self.routes.layers.items())):
            width = geometry["experts"] // self.size
            begin, end = self.rank * width, (self.rank + 1) * width
            for identity, layers in self.routes.assignments.items():
                for first, last, experts in layers[layer]:
                    receiving = any(begin <= expert < end for expert in experts)
                    if first < prompts[identity]:
                        conditions.add((local, "prefill", receiving))
                    if last > prompts[identity]:
                        conditions.add((local, "decode", receiving))

            # Unused table positions cannot require observing an additional receiver.
            for first, _, experts in self.routes.padding_routes.get(layer, ()):
                if first < self.routes.padding_limit:
                    conditions.add((local, "padding", any(begin <= expert < end for expert in experts)))
        return sorted(conditions)

    def bind_batch(self, parameters: dict[str, int], members) -> dict:
        """Bind local expert rows and return the peer inputs for this invocation."""
        model_tokens = parameters["model_tokens"]
        self.routes.check_batch(members, model_tokens)
        if self.size == 1:

            # Local dispatch keeps token rows; expert expansion happens inside the computation scope.
            return {}
        if model_tokens % self.size:
            raise ValueError("DeepEP input must be padded before splitting tokens across its group")

        traffic = {}
        for layer, key in self.layer_parameters.items():
            sent, received = self.combine_counts(members, model_tokens, layer)

            # Combine sends back one row for every expert selected on this worker.
            parameters[key] = sum(sent)
            traffic[key] = {"send": sent, "receive": received}
        return {"expert_traffic": traffic}

    def combine_counts(self, members, model_tokens, layer):
        width = self.routes.layers[layer]["experts"] // self.size
        rows = model_tokens // self.size
        sent, received = [0] * self.size, [0] * self.size
        position = 0
        for count, experts in self.routes.batch_segments(members, layer, model_tokens):
            receivers = Counter(expert // width for expert in experts)
            end = position + count

            # Dispatch partitions padded token order; Combine reverses exactly those peer-wise transfers.
            for sender in range(position // rows, (end - 1) // rows + 1):
                overlap = min(end, (sender + 1) * rows) - max(position, sender * rows)
                if sender == self.rank:
                    for receiver, copies in receivers.items():
                        received[receiver] += overlap * copies
                sent[sender] += overlap * receivers[self.rank]
            position = end
        return sent, received

    def scope(self, enclosing, layer: int | None):
        parameters = {name: value for name, value in enclosing["parameters"].items() if not name.startswith("expert/")}
        arguments = {name: name for name in parameters if name not in enclosing["constants"]}
        if layer is not None:
            key = self.layer_parameters[layer]
            if key in enclosing["parameters"]:
                parameters["expert_received_tokens"] = enclosing["parameters"][key]
                arguments["expert_received_tokens"] = key
        return parameters, arguments


@dataclass(frozen=True)
class MhaKvLayout:
    start_layer: int
    end_layer: int
    head_num: int
    dtype: str
    key_token_bytes: int
    value_token_bytes: int

    @classmethod
    def from_pool(cls, pool):
        token_layout = device_runtime().kv_token_layout(pool.k_buffer, pool.v_buffer)
        return cls(pool.start_layer, pool.start_layer + pool.layer_num, pool.head_num, *token_layout)

    @classmethod
    def read(cls, value):
        layout = cls(**value)
        if not isinstance(layout.dtype, str) or any(
            type(field) is not int for name, field in value.items() if name != "dtype"
        ):
            raise TypeError("KV layout requires a dtype name and integer extents")
        if (
            not 0 <= layout.start_layer < layout.end_layer
            or min(layout.head_num, layout.key_token_bytes, layout.value_token_bytes) <= 0
        ):
            raise ValueError("KV layout requires an ordered layer interval and positive head/byte extents")
        if layout.key_token_bytes % layout.head_num or layout.value_token_bytes % layout.head_num:
            raise ValueError("KV token bytes must contain whole heads")
        return layout

    def document(self):
        return asdict(self)

    def buffers(self, token_capacity, page_size):
        layers = self.end_layer - self.start_layer
        widths = [self.key_token_bytes] * layers + [self.value_token_bytes] * layers
        lengths = [width * ((token_capacity // page_size + 1) * page_size) for width in widths]

        # These are disjoint coordinates for native address arithmetic, never process or device pointers.
        starts = list(accumulate(lengths, initial=1))[:-1]
        return KvBufferAddresses(self, page_size, starts, lengths, [width * page_size for width in widths])


class KvBufferAddresses:
    def __init__(self, layout, page_size, starts, lengths, item_bytes):
        self.layout = layout
        self.page_size = page_size
        layers = layout.end_layer - layout.start_layer
        self.token_bytes = [layout.key_token_bytes] * layers + [layout.value_token_bytes] * layers
        self.starts, self.lengths, self.item_bytes = starts, lengths, item_bytes
        if (
            len(starts) != len(self.token_bytes)
            or len(lengths) != len(starts)
            or item_bytes != [width * page_size for width in self.token_bytes]
        ):
            raise ValueError("registered KV buffers differ from the observed layer/page layout")
        self.addresses = sorted((address, index) for index, address in enumerate(starts))
        previous_end = 0
        for address, index in self.addresses:
            length = lengths[index]
            if address < previous_end or length <= 0 or length % item_bytes[index]:
                raise ValueError("registered KV buffers overlap or contain partial pages")
            previous_end = address + length

    def native_infos(self):
        return self.starts, self.lengths, self.item_bytes

    def locate(self, address, length):
        position = bisect_right(self.addresses, (address, len(self.starts))) - 1
        if position < 0:
            raise ValueError("native KV transfer precedes all registered buffers")
        index = self.addresses[position][1]
        if length < 0 or address + length > self.starts[index] + self.lengths[index]:
            raise ValueError("native KV transfer crosses an observed buffer boundary")
        return index, address - self.starts[index], length

    def regions(self, address, length):
        index, offset, length = self.locate(address, length)
        layers = self.layout.end_layer - self.layout.start_layer
        width = self.token_bytes[index]
        component = 0 if index < layers else self.layout.key_token_bytes
        token_bytes = self.layout.key_token_bytes + self.layout.value_token_bytes
        regions = []
        stop = offset + length

        # Only partial first/last rows split; any intervening full rows stay one range.
        while offset < stop:
            token, begin = divmod(offset, width)
            remaining = stop - offset
            if begin:
                tokens, end = 1, min(width, begin + remaining)
            elif remaining >= width:
                tokens, end = remaining // width, width
            else:
                tokens, end = 1, remaining
            regions.append({
                "layer": self.layout.start_layer + index % layers,
                "token_begin": token, "token_end": token + tokens,
                "slice": {"token_bytes": token_bytes, "begin": component + begin, "end": component + end},
            })
            offset += (end - begin) * tokens
        return regions
