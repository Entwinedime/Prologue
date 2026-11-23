"""Input fields interpreted by the SGLang integration."""

from dataclasses import dataclass


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
