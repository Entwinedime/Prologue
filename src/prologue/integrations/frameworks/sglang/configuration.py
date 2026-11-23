"""Input fields interpreted by the SGLang integration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Model:
    path: str
    dtype: str
    load_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("model path must be a nonempty string")
        if self.dtype not in ("bfloat16", "float16") or self.load_format != "dummy":
            raise ValueError("SGLang inputs require bfloat16 or float16 and dummy weights")


@dataclass(frozen=True, slots=True)
class Configuration:
    id: str
    tp: int
    attention_backend: str
    page_size: int
    max_running_requests: int
    max_total_tokens: int
    max_prefill_tokens: int
    pp: int = 1
    microbatch_requests: int | None = None
    ep: int = 1
    prefill_chunk_tokens: int = 0
    mixed_batches: bool = False
    prefix_cache: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("configuration ID must be a nonempty string")
        if not isinstance(self.attention_backend, str) or not self.attention_backend:
            raise ValueError(f"{self.id} needs an attention backend")

        limits = ("tp", "pp", "ep", "page_size", "max_running_requests", "max_total_tokens", "max_prefill_tokens")
        if self.microbatch_requests is not None:
            if self.pp == 1:
                raise ValueError(f"{self.id} microbatch_requests requires pipeline parallelism")
            limits += ("microbatch_requests",)
        for field in limits:
            value = getattr(self, field)
            if type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(f"{self.id}/{field} must be a positive signed 64-bit integer")
        if self.max_total_tokens < self.page_size:
            raise ValueError(f"{self.id} KV capacity must hold one page")
        if self.ep not in (1, self.tp):
            raise ValueError(f"{self.id} native DeepEP requires expert width equal to tensor width")
        chunk = self.prefill_chunk_tokens
        if type(chunk) is not int or not 0 <= chunk < 1 << 63 or chunk % self.page_size:
            raise ValueError(f"{self.id} prefill_chunk_tokens must be a nonnegative multiple of page_size")
        if type(self.mixed_batches) is not bool or (self.mixed_batches and chunk == 0):
            raise ValueError(f"{self.id} mixed_batches requires a positive Prefill chunk size")
        if type(self.prefix_cache) is not bool or (self.prefix_cache and (self.pp > 1 or self.ep > 1)):
            raise ValueError(f"{self.id} prefix_cache requires a boolean and a single-stage dense instance")

    @property
    def worker_count(self):
        return self.tp * self.pp

    def route_table(self, workload):
        if self.pp > 1 and ("prefixes" in workload or any("prefix_id" in row for row in workload["requests"])):
            raise ValueError(f"{self.id} shared prompt content requires a single-stage instance")
        routes = workload.get("expert_routes", {})
        if not isinstance(routes, dict):
            raise ValueError(f"{workload['id']} expert_routes must be an object keyed by EP width")
        width = str(self.ep)
        if width in routes:
            if self.prefix_cache:
                raise ValueError(f"{self.id} prefix caching requires dense cache keys")
            if not isinstance(routes[width], dict):
                raise ValueError(f"{workload['id']} requires an expert route table for {self.id}")
            return routes[width]
        if self.ep > 1:
            raise ValueError(f"{workload['id']} requires expert routes for {self.id}")
        return None

    def document(self):
        fields = asdict(self)
        if self.pp == 1:
            del fields["pp"]
        if self.microbatch_requests is None:
            del fields["microbatch_requests"]
        if self.ep == 1:
            del fields["ep"]
        if not self.prefill_chunk_tokens:
            del fields["prefill_chunk_tokens"]
        if not self.mixed_batches:
            del fields["mixed_batches"]
        if not self.prefix_cache:
            del fields["prefix_cache"]
        return fields


def scheduling_options(configuration):
    return {"chunked_prefill_size": configuration.get("prefill_chunk_tokens", 0) or -1,
            "enable_mixed_chunk": configuration.get("mixed_batches", False),
            "disable_radix_cache": not configuration.get("prefix_cache", False)}


class Disaggregation:
    roles = ("prefill", "decode")

    def __init__(self, id, attention_backend, page_size, disaggregation):
        if not isinstance(disaggregation, dict) or disaggregation.keys() != set(self.roles):
            raise ValueError("disaggregation requires distinct prefill and decode configurations")
        self.id = id
        self.instances = {}
        for role in self.roles:
            settings = disaggregation[role]
            if not isinstance(settings, dict) or settings.keys() != {
                "tp", "max_running_requests", "max_total_tokens", "max_prefill_tokens",
            }:
                raise ValueError(f"{id}/{role} requires tp and independent request, KV and prefill limits")
            self.instances[role] = Configuration(
                id, attention_backend=attention_backend, page_size=page_size, **settings,
            )
        if self.instances["prefill"].tp != 1:
            raise ValueError(f"{id} PD handoff evidence currently requires Prefill TP1")

    def route_table(self, workload):
        if "expert_routes" in workload:
            raise ValueError(f"{self.id} requires a workload for independently placed dense workers")
        if "prefixes" in workload or any("prefix_id" in request for request in workload["requests"]):
            raise ValueError(f"{self.id} shared prompt content requires a colocated instance")
        return None

    @staticmethod
    def native_policy(role):
        return {
            "disaggregation_mode": role, "disaggregation_transfer_backend": "ascend",
            "num_reserved_decode_tokens": 32, "disaggregation_decode_polling_interval": 8,
            "disaggregation_decode_extra_slots": 4,
        }

    def resource_pools(self):
        return [
            {"slots": configuration.max_running_requests + (
                self.native_policy(role)["disaggregation_decode_extra_slots"] if role == "decode" else 0),
             "kv_tokens": configuration.max_total_tokens, "sort_released_pages": True}
            for role, configuration in self.instances.items()
        ]

    def document(self):
        options = {role: configuration.document() for role, configuration in self.instances.items()}
        shared = {"id", "attention_backend", "page_size"}
        return {
            **{key: options["prefill"][key] for key in shared},
            "disaggregation": {role: {key: value for key, value in fields.items() if key not in shared}
                               for role, fields in options.items()},
        }


def configuration_from(document):
    if "disaggregation" in document:
        return Disaggregation(**document)
    return Configuration(**document)


def native_options(model, configuration):
    config = configuration.document()
    del config["id"]
    schedule = scheduling_options(config)
    config.pop("prefill_chunk_tokens", None)
    config.pop("mixed_batches", None)
    config.pop("prefix_cache", None)
    config["tp_size"] = config.pop("tp")
    config["pp_size"] = config.pop("pp", 1)
    if "microbatch_requests" in config:
        config["pp_max_micro_batch_size"] = config.pop("microbatch_requests")
    if configuration.ep > 1:
        config.update(ep_size=config.pop("ep"), moe_a2a_backend="deepep", deepep_mode="normal")
    weights = asdict(model)
    weights["model_path"] = weights.pop("path")
    return {
        **weights, **config, **schedule, "device": "npu", "skip_tokenizer_init": True,
        "disable_overlap_schedule": True, "disable_cuda_graph": True,
        "stream_interval": 1, "mem_fraction_static": 0.8, "log_level": "warning",
    }
