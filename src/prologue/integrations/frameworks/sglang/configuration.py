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

    @property
    def worker_count(self):
        return self.tp * self.pp

    def route_table(self, workload):
        routes = workload.get("expert_routes", {})
        if not isinstance(routes, dict):
            raise ValueError(f"{workload['id']} expert_routes must be an object keyed by EP width")
        width = str(self.ep)
        if width in routes:
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
        return fields


def native_options(model, configuration):
    config = configuration.document()
    del config["id"]
    config["tp_size"] = config.pop("tp")
    config["pp_size"] = config.pop("pp", 1)
    if "microbatch_requests" in config:
        config["pp_max_micro_batch_size"] = config.pop("microbatch_requests")
    if configuration.ep > 1:
        config.update(ep_size=config.pop("ep"), moe_a2a_backend="deepep", deepep_mode="normal")
    weights = asdict(model)
    weights["model_path"] = weights.pop("path")
    return {
        **weights, **config, "device": "npu", "skip_tokenizer_init": True,
        "disable_overlap_schedule": True, "disable_cuda_graph": True,
        "disable_radix_cache": True, "chunked_prefill_size": -1, "stream_interval": 1,
        "mem_fraction_static": 0.8, "log_level": "warning",
    }
