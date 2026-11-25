"""Read native batch identity and token ranges without tensor-operation observation."""

from __future__ import annotations

from typing import Any

from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols


class BatchTensors:
    @staticmethod
    def bind_location_rows(scope, slots, writes):
        members = scope["work"]["members"]
        owners = {slot: member["id"] for slot, member in zip(slots, members, strict=True)}
        if len(owners) != len(members):
            raise ValueError("admitted members share a request-table slot")
        output, = (port["tensor"] for port in scope["ports"]
                   if port["name"] == "cache_locations" and port["direction"] == "output")

        # The native row selects its request. The template keeps the packing rule, independently of physical slots.
        output["packing"] = {
            "table": "request_to_token",
            "member_count": "members", "row_extent": "sequence_capacity",
            "range": {"begin": "context_tokens", "end": ["add", "context_tokens", "new_tokens"]},
            "slices": [{"request": owners[row["slot"]], "source": row["source"], "target": row["target"]}
                       for row in writes],
        }

    @staticmethod
    def boundaries(batch: Any) -> dict[str, tuple[Any, list[str]]]:
        # Capture and control publish the same batch ABI; native scratch tensors stay inside the call.
        return {
            "request_slots": (batch.req_pool_indices, ["members"]),
            "sequence_lengths": (batch.seq_lens, ["members"]),
            "cache_locations": (batch.out_cache_loc, ["input_tokens"]),
        }

    @staticmethod
    def values(batch: Any) -> dict[str, Any]:
        return {name: tensor for name, (tensor, _) in BatchTensors.boundaries(batch).items()}

    @staticmethod
    def phase(batch: Any) -> str:
        if batch.forward_mode.is_mixed():
            return "mixed"
        return "prefill" if batch.forward_mode.is_extend() or batch.forward_mode.is_prebuilt() else "decode"

    @staticmethod
    def members(batch: Any) -> list[dict[str, Any]]:
        sequences = batch.seq_lens_cpu.tolist()
        if batch.forward_mode.is_extend() or batch.forward_mode.is_prebuilt():
            lengths = batch.extend_lens
        else:
            lengths = [1] * len(batch.reqs)
        decoding = {request.rid for request in batch.decoding_reqs or ()}
        return [
            {
                "id": request.rid, "new_tokens": int(tokens), "context_tokens": int(sequence - tokens),
                "slot": int(request.req_pool_idx),
                "phase": "decode" if batch.forward_mode.is_decode() or request.rid in decoding else "prefill",
            }
            for request, sequence, tokens in zip(batch.reqs, sequences, lengths, strict=True)
        ]

    @staticmethod
    def input_dimensions(tokens: int, members: int) -> dict[str, int]:
        return {"input_tokens": int(tokens), "members": int(members)}

    @staticmethod
    def parameters(batch: Any, tokens: int) -> dict[str, int]:
        dimensions = BatchTensors.input_dimensions(tokens, len(batch.reqs))
        return {
            **BatchTensors.capacity(batch.token_to_kv_pool_allocator, batch.req_to_token_pool),
            **dimensions,
            "model_tokens": dimensions["input_tokens"], "model_members": dimensions["members"],
        }

    @staticmethod
    def capacity(pool, slots):
        mapping = slots.req_to_token
        return {
            "kv_pages": pool.size // pool.page_size, "page_size": pool.page_size,
            "slot_capacity": mapping.shape[0] - 1, "sequence_capacity": mapping.shape[1],
        }

    @staticmethod
    def align_model_tokens(tokens: int, alignment: int) -> int:
        return (tokens + alignment - 1) // alignment * alignment

    @staticmethod
    def model_parameters(batch: Any, tokens: int) -> dict[str, int]:
        model_tokens, model_members = tokens, len(batch.reqs)
        if batch.global_num_tokens is not None:
            get_parallel = resolve(symbols().get_parallel)

            # Native eager MLP-sync pads token rows and, for Decode, the model's batch metadata as well.
            parallel = get_parallel()
            if len(batch.global_num_tokens) != 1 or parallel.attn_cp_size != 1:
                raise ValueError("model padding requires a declared DP/CP layout")
            model_tokens = BatchTensors.align_model_tokens(batch.global_num_tokens[0], parallel.attn_tp_size)
            if batch.forward_mode.is_decode():
                model_members = model_tokens
        return {"model_tokens": model_tokens, "model_members": model_members}

    @staticmethod
    def model_padding_guards(parameters: dict[str, int]) -> list[list[str]]:
        # Padding zero rows and materializing a positive tail run different native work.
        dimensions = (("input_tokens", "model_tokens"), ("members", "model_members"))
        return [["eq" if parameters[actual] == parameters[padded] else "lt", actual, padded]
                for actual, padded in dimensions]

    @staticmethod
    def attention_layout(backend, parameters, members):
        if backend not in platform_domain().strided_attention:
            return {}

        # This backend indexes a dense [batch, longest] table, then takes every page_size-th column.
        longest = max((member["context_tokens"] + member["new_tokens"] for member in members), default=0)
        page = parameters["page_size"]
        strided = page > 1 and longest > 1 and (parameters["model_members"] > 1 or longest > page)
        return {"block_table_strided": int(strided)}
