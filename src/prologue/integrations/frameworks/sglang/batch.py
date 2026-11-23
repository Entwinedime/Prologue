"""Read the actual token boundaries and output semantics of a native batch."""


class BatchWork:
    @staticmethod
    def phase(mode):
        if mode.is_mixed():
            return "mixed"
        if mode.is_decode():
            return "decode"
        if mode.is_extend():
            return "prefill"
        raise ValueError(f"unsupported native batch mode {mode}")

    @classmethod
    def members(cls, batch, identities, slot_offset=0):
        mode = batch.forward_mode
        begins = batch.prefix_lens if mode.is_extend() else None
        decoding = set(batch.decoding_reqs or ())
        members = []
        for index, (request, identity) in enumerate(zip(batch.reqs, identities, strict=True)):
            end = request.kv_committed_len
            member = {"request": identity, "slot": int(request.req_pool_idx) - slot_offset,
                      "kv_committed_tokens": end, "kv_reserved_tokens": request.kv_allocated_len,
                      "output_tokens": len(request.output_ids)}

            # A pure Decode row has one new token. An Extend row can start inside the prompt.
            if mode.is_mixed():
                member["kind"] = "decode" if request in decoding else "prefill"
                member["read_tokens"] = begins[index]
            elif begins is not None and begins[index]:
                member["read_tokens"] = begins[index]
            if request.inflight_middle_chunks:
                member["publishes"] = False
            members.append(member)
        return members

    @classmethod
    def attention(cls, forward_batch):
        phase = cls.phase(forward_batch.forward_mode)
        parameters = {"phase": phase, "seq_lens": forward_batch.seq_lens_cpu.tolist()}
        if forward_batch.forward_mode.is_extend():
            begins = forward_batch.extend_prefix_lens_cpu
            if any(begins):
                parameters["read_tokens"] = list(begins)
        return parameters

    @staticmethod
    def cache(batch, members):
        end = max(member["kv_committed_tokens"] for member in members)
        table = batch.req_to_token_pool.req_to_token
        rows = table[batch.req_pool_indices, :end].detach().cpu().tolist()
        return {
            "rows": [row[:member["kv_committed_tokens"]] for row, member in zip(rows, members, strict=True)],
            "written": batch.out_cache_loc.detach().cpu().tolist(),
        }
