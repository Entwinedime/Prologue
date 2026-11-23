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

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("configuration ID must be a nonempty string")
        if not isinstance(self.attention_backend, str) or not self.attention_backend:
            raise ValueError(f"{self.id} needs an attention backend")

        limits = ("tp", "page_size", "max_running_requests", "max_total_tokens", "max_prefill_tokens")
        for field in limits:
            value = getattr(self, field)
            if type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(f"{self.id}/{field} must be a positive signed 64-bit integer")
        if self.max_total_tokens < self.page_size:
            raise ValueError(f"{self.id} KV capacity must hold one page")


def native_options(model, configuration):
    config = asdict(configuration)
    del config["id"]
    config["tp_size"] = config.pop("tp")
    weights = asdict(model)
    weights["model_path"] = weights.pop("path")
    return {
        **weights, **config, "device": "npu", "skip_tokenizer_init": True,
        "disable_overlap_schedule": True, "disable_cuda_graph": True,
        "disable_radix_cache": True, "chunked_prefill_size": -1, "stream_interval": 1,
        "mem_fraction_static": 0.8, "log_level": "warning",
    }
