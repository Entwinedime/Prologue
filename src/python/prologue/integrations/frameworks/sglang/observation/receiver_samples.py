"""Build and observe a native prebuilt batch from the receiver's own pools."""

from array import array
from dataclasses import dataclass
from time import monotonic_ns

import torch

from prologue.common.io import write_json
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.call_ports import (
    LocalModel,
    NativeCallObservation,
)
from prologue.integrations.frameworks.sglang.observation.serving import FlowScopes
from prologue.integrations.frameworks.sglang.observation.window import (
    LocalProbe,
    publish_facts,
)
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors
from prologue.integrations.frameworks.sglang.shared.disaggregated_calls import DisaggregatedCalls
from prologue.integrations.frameworks.sglang.shared.kv_layout import MhaKvLayout


@dataclass(frozen=True, slots=True)
class ReceiverScheduler:
    server_args: object
    enable_hisparse: bool = False


class ReceiverPools:
    def __init__(self, job):
        DecodePreallocQueue = resolve(symbols().decode_prealloc_queue)
        ChunkCache = resolve(symbols().chunk_cache)

        domain, cap = job["domain"], job["capacity"]
        impl = domain["implementation"]
        page = cap["page_size"]
        size = cap["kv_pages"] * page
        loaded = LocalModel(domain, page, disaggregation_mode="decode",
                            disaggregation_transfer_backend=platform_domain().pd_transfer, disable_radix_cache=True)
        settings, self.model = loaded.args, loaded.config

        if not loaded.matches:
            model = loaded.document
            differences = [key for key in model.keys() | impl["model"].keys()
                           if model.get(key) != impl["model"].get(key)]
            raise ValueError(f"receiver model mismatch: fields {sorted(differences)}, dtype {self.model.dtype}")

        layout = MhaKvLayout.read(domain["kv_layout"])
        dtype = getattr(torch, layout.dtype.removeprefix("torch."))
        width = dtype.itemsize * layout.head_num
        self.kv = resolve(symbols().mha_pool)(size, page, dtype, layout.head_num, layout.key_token_bytes // width,
                                      layout.end_layer - layout.start_layer, platform_domain().device_type, False,
                                      v_head_dim=layout.value_token_bytes // width,
                                      start_layer=layout.start_layer, end_layer=layout.end_layer)

        if MhaKvLayout.from_pool(self.kv) != layout:
            raise ValueError("allocated KV layout differs from the requested receiver")
        self.allocator = resolve(symbols().paged_allocator)(
            size, page, torch.int64, platform_domain().device_type, self.kv, False
        )

        # The batch is already admitted. Preparation only sees the native table's total row count.
        self.slots = resolve(symbols().decode_req_to_token_pool)(
            cap["slot_capacity"], cap["sequence_capacity"], platform_domain().device_type, False, 0
        )
        self.cache = ChunkCache(resolve(symbols().cache_init_params)(True, self.slots, self.allocator, page))

        # Bind the production reservation operation without starting its transport queue or a remote scheduler.
        self.reserve = DecodePreallocQueue.__new__(DecodePreallocQueue)
        self.reserve.scheduler = ReceiverScheduler(settings)
        self.reserve.req_to_token_pool = self.slots
        self.reserve.token_to_kv_pool_allocator = self.allocator
        self.reserve.token_to_kv_pool = self.kv

    def batch(self, members):
        Req = resolve(symbols().req)
        ScheduleBatch = resolve(symbols().schedule_batch)
        SamplingParams = resolve(symbols().sampling_params)
        SpeculativeAlgorithm = resolve(symbols().speculative_algorithm)

        requests, locations = [], []
        for i, member in enumerate(members, start=1):
            ids = array("q", (n % self.model.vocab_size for n in range(member["new_tokens"])))
            request = Req(member["id"], "", ids, SamplingParams(temperature=0, max_new_tokens=2))
            loc = self.reserve._pre_alloc(request)

            # KV delivery precedes this cut. Seed distinct real bytes so the final table can be checked by a consumer.
            self.kv.k_buffer.flatten(1, 2).index_fill_(1, loc, i)
            self.kv.v_buffer.flatten(1, 2).index_fill_(1, loc, -i)
            request.output_ids.append(ids[-1])
            request.init_next_round_input(self.cache)
            request.set_extend_range(len(request.prefix_indices), request.kv_committed_len)
            requests.append(request)
            locations.append(loc.tolist())

        batch = ScheduleBatch.init_new(requests, self.slots, self.allocator, self.cache, self.model,
                                       False, SpeculativeAlgorithm.NONE)
        return batch, locations

    def readable(self, batch, locations):
        flat = [loc for row in locations for loc in row]
        if batch.out_cache_loc.tolist() != flat or min(flat) <= 0 or len(set(flat)) != len(flat):
            return False
        for tag, loc in enumerate(batch.out_cache_loc.split([len(row) for row in locations]), start=1):
            for buffer, value in ((self.kv.k_buffer, tag), (self.kv.v_buffer, -tag)):
                if not (buffer.flatten(1, 2).index_select(1, loc) == value).all().item():
                    return False
        return True


class PrebuiltRunner(LocalProbe):
    def __init__(self, job, output):
        super().__init__(job, output)
        self.rank = job["rank"]

    def observe(self, pools):
        ScheduleBatch = resolve(symbols().schedule_batch)

        batches = [(members, *pools.batch(members)) for members in self.job["batches"]]
        markers = device_runtime().markers(detailed_timing=False)
        owner = NativeCallObservation(markers, "decode")
        owner.active = True
        owner.parallel = {**self.job["domain"]["parallel"], "rank": self.rank}
        owner.aliases = {m["id"]: m["id"] for members, _, _ in batches for m in members}
        cuts = FlowScopes(owner)
        markers.synchronize()

        with device_runtime().scoped_tasks.profile(self.output, self.rank), markers.window():
            for _, batch, _ in batches:
                DisaggregatedCalls(cuts).prepare(ScheduleBatch.prepare_for_prebuilt, batch)

        # The final fence and KV reads belong to qualification, outside the captured asynchronous calls.
        finished = device_runtime().api.Event()
        finished.record()
        finished.synchronize()
        environment = {**self.job["domain"], **markers.environment(), "parallel": owner.parallel}
        raw = owner.document(environment, cuts.queries)

        # The local qualifier consumes these physical writes alongside the independent device readback.
        observed = [scope.pop("page_table") for scope in raw["scopes"] if "page_table" in scope]
        write_json(self.output / f"rank-{self.rank}.json", raw)
        markers.release()
        rows = []

        for (members, batch, locations), table in zip(batches, observed, strict=True):
            if table["slots"] != batch.req_pool_indices.tolist():
                raise ValueError("prepared batch slots differ from the native source rows")
            rows.append({"members": members, "request_slots": table["slots"],
                         "parameters": BatchTensors.parameters(batch, batch.extend_num_tokens),
                         "writes": table["writes"], "verified_kv": pools.readable(batch, locations),
                         "completion": "device_event_synchronized_before_consumer_read"})

        write_json(self.output / "receiver.json", rows)
        return raw

    def run(self, revision):
        ScheduleBatch = resolve(symbols().schedule_batch)

        started = monotonic_ns()
        device_runtime().api.set_device(0)
        pools = ReceiverPools(self.job)
        initialized = monotonic_ns()
        warmup, _ = pools.batch(self.job["batches"][0])
        ScheduleBatch.prepare_for_prebuilt(warmup)
        device_runtime().markers.synchronize()
        pools.allocator.clear()
        pools.slots.clear()
        warmed = monotonic_ns()
        raw = self.observe(pools)

        exporting = monotonic_ns()
        publish_facts(self.output, self.rank, raw, self.output / "sdk", "facts.json")
        self.publish(
            revision, 0,
            measurements='receiver.json',
            facts='facts.json',
            initialization_ns=initialized - started,
            warmup_ns=warmed - initialized,
            capture_ns=exporting - warmed,
            postprocessing_ns=monotonic_ns() - exporting,
        )
