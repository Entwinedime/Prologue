"""Bind declared expert choices to the token intervals of a native batch."""


class ExpertRoutes:
    def __init__(self, source, requests, layers, layer_count):
        if not isinstance(source, dict) or not {"assignments"} <= source.keys() <= {"assignments", "padding"}:
            raise ValueError("expert routes require assignments and accept optional padding")
        assignments = source["assignments"]
        padding = source.get("padding", [])
        if not isinstance(assignments, list) or not isinstance(padding, list):
            raise ValueError("expert assignments and padding must be arrays")
        self.layers = sorted(layers)
        self.layer_count = layer_count
        lengths = {row["id"]: row["prompt_tokens"] + row["output_tokens"] - 1 for row in requests}
        self.requests = {name: {} for name in lengths}
        self.padding = {}

        for row in assignments:
            if not isinstance(row, dict) or row.keys() != {"request", "layers", "tokens", "experts"}:
                raise ValueError("expert assignment fields are request, layers, tokens and experts")
            request = row["request"]
            if request not in lengths:
                raise ValueError(f"expert assignment names an unknown request {request}")
            self.add(self.requests[request], row, lengths[request])
        for request, length in lengths.items():
            self.coverage(self.requests[request], length, request)

        padding_limit = 0
        for row in padding:
            if not isinstance(row, dict) or row.keys() != {"layers", "tokens", "experts"}:
                raise ValueError("padding assignment fields are layers, tokens and experts")
            padding_limit = max(padding_limit, self.add(self.padding, row, (1 << 63) - 1))
        self.coverage(self.padding, padding_limit, "padding")
        self.padding_limit = padding_limit

    def require_geometry(self, geometry):
        sources = [*self.requests.values(), self.padding]
        for layer, (count, top_k) in geometry.items():
            for table in sources:
                for _, _, experts in table.get(layer, ()):
                    if len(experts) != top_k or max(experts) >= count:
                        raise ValueError(f"expert choices exceed the native geometry of layer {layer}")

    @staticmethod
    def interval(value, limit, field):
        if (not isinstance(value, list) or len(value) != 2 or any(type(part) is not int for part in value)
                or not 0 <= value[0] < value[1] <= limit):
            raise ValueError(f"expert {field} requires a bounded integer [begin, end) interval")
        return value

    def add(self, table, row, length):
        first, last = self.interval(row["layers"], self.layer_count, "layers")
        begin, end = self.interval(row["tokens"], length, "tokens")
        experts = row["experts"]
        if (
            not isinstance(experts, list) or not experts
            or any(type(value) is not int or value < 0 for value in experts)
            or len(set(experts)) != len(experts)
        ):
            raise ValueError("expert choices must be distinct nonnegative integer IDs")
        for layer in self.layers:
            if first <= layer < last:
                table.setdefault(layer, []).append((begin, end, experts))
        return end

    def coverage(self, table, length, owner):
        for layer in self.layers:
            intervals = table.get(layer, [])
            intervals.sort(key=lambda row: row[0])
            covered = 0
            for begin, end, _ in intervals:
                if begin != covered:
                    raise ValueError(f"expert route gap or overlap for {owner}, layer {layer}")
                covered = end
            if covered != length:
                raise ValueError(f"incomplete expert routes for {owner}, layer {layer}")

    @staticmethod
    def append(output, intervals, begin, end):
        if not 0 <= begin < end <= intervals[-1][1]:
            raise ValueError("native batch token interval exceeds declared expert routes")
        for first, last, experts in intervals:
            count = min(last, end) - max(first, begin)
            if count > 0:
                if output and output[-1][1] == experts:
                    output[-1][0] += count
                else:
                    output.append([count, experts])

    def bind(self, members, layer, rows):
        tokens = sum(end - begin for _, begin, end in members)
        padding = rows - tokens
        if not 0 <= padding <= self.padding_limit:
            raise ValueError(f"native layer {layer} requires {padding} expert padding rows for {tokens} tokens")

        choices = []
        for request, begin, end in members:
            self.append(choices, self.requests[request][layer], begin, end)
        if padding:
            self.append(choices, self.padding[layer], 0, padding)
        return {"layer": layer, "tokens": tokens, "padding": padding, "choices": choices}

    @staticmethod
    def members(batch, aliases=None):
        lengths = batch.extend_lens if batch.forward_mode.is_extend() else [1] * len(batch.reqs)
        return [(row.rid if aliases is None else aliases[row.rid], end - length, end)
                for row, end, length in zip(batch.reqs, batch.seq_lens_cpu.tolist(), lengths, strict=True)]

    @staticmethod
    def native_modules(model):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return [(name, module) for name, module in model.named_modules() if isinstance(module, FusedMoE)]

    def aligned(self, batch, alignment):
        members = self.members(batch)
        tokens = sum(end - begin for _, begin, end in members)

        # Native DeepEP partitions an aligned logical batch across equally sized expert input shards.
        rows = (tokens + alignment - 1) // alignment * alignment
        return [self.bind(members, layer, rows) for layer in self.layers]

    @staticmethod
    def shard(binding, begin, end):
        offset = 0
        rows = []
        for length, experts in binding["choices"]:
            count = min(offset + length, end) - max(offset, begin)
            if count > 0:
                rows.extend([experts] * count)
            offset += length
        return rows
