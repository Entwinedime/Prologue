"""Run SGLang's PD receiver preparation against a populated local page table."""

import json
from array import array
from time import monotonic_ns
from types import SimpleNamespace

import torch

from prologue.runtime.io import write_json
from prologue.integrations.frameworks.sglang.batches import BatchTensors
from prologue.integrations.frameworks.sglang.disaggregated_calls import DisaggregatedCalls
from prologue.integrations.frameworks.sglang.kv_layout import MhaKvLayout
from prologue.integrations.frameworks.sglang.observation.call_scopes import NativeCallObservation
from prologue.integrations.frameworks.sglang.observation.flow_scopes import FlowScopes
from prologue.integrations.frameworks.sglang.observation.source_scopes import SGLangSourceScopes
from prologue.integrations.platforms.ascend.export import AscendExport
from prologue.integrations.platforms.ascend.markers import AscendMarkers
from prologue.integrations.platforms.ascend.observation import AscendObservation
from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks


class ReceiverPools:
    def __init__(self, job):
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeReqToTokenPool
        from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

        domain, cap = job["domain"], job["capacity"]
        impl = domain["implementation"]
        page = cap["page_size"]
        size = cap["kv_pages"] * page
        settings = ServerArgs(model_path=impl["model"]["_name_or_path"], dtype=impl["dtype"].removeprefix("torch."),
                              device="npu", tp_size=domain["parallel"]["logical_tp"], page_size=page,
                              attention_backend=impl["attention_backend"], load_format=impl["load_format"],
                              skip_tokenizer_init=True,
                              disaggregation_mode="decode", disaggregation_transfer_backend="ascend",
                              disable_overlap_schedule=True, disable_cuda_graph=True, disable_radix_cache=True)
        set_global_server_args_for_scheduler(settings)
        self.model = ModelConfig.from_server_args(settings)
        model = json.loads(self.model.hf_config.to_json_string(use_diff=False))
        if model != impl["model"] or str(self.model.dtype) != impl["dtype"]:
            differences = [key for key in model.keys() | impl["model"].keys()
                           if model.get(key) != impl["model"].get(key)]
            raise ValueError(f"receiver model mismatch: fields {sorted(differences)}, dtype {self.model.dtype}")

        layout = MhaKvLayout.read(domain["kv_layout"])
        dtype = getattr(torch, layout.dtype.removeprefix("torch."))
        width = dtype.itemsize * layout.head_num
        self.kv = NPUMHATokenToKVPool(size, page, dtype, layout.head_num, layout.key_token_bytes // width,
                                      layout.end_layer - layout.start_layer, "npu", False,
                                      v_head_dim=layout.value_token_bytes // width,
                                      start_layer=layout.start_layer, end_layer=layout.end_layer)
        if MhaKvLayout.from_pool(self.kv) != layout:
            raise ValueError("allocated KV layout differs from the requested receiver")
        self.allocator = NPUPagedTokenToKVPoolAllocator(size, page, torch.int64, "npu", self.kv, False)

        # The batch is already admitted. Preparation only sees the native table's total row count.
        self.slots = DecodeReqToTokenPool(cap["slot_capacity"], cap["sequence_capacity"], "npu", False, 0)
        self.cache = ChunkCache(CacheInitParams(True, self.slots, self.allocator, page))

        # Bind the production reservation operation without starting its transport queue or a remote scheduler.
        self.reserve = DecodePreallocQueue.__new__(DecodePreallocQueue)
        self.reserve.scheduler = SimpleNamespace(server_args=settings, enable_hisparse=False)
        self.reserve.req_to_token_pool = self.slots
        self.reserve.token_to_kv_pool_allocator = self.allocator
        self.reserve.token_to_kv_pool = self.kv

    def batch(self, members):
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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


class PrebuiltRunner:
    def __init__(self, job, output):
        self.job, self.output = job, output
        self.rank = job["rank"]

    def observe(self, pools):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        batches = [(members, *pools.batch(members)) for members in self.job["batches"]]
        markers = AscendMarkers(detailed_timing=False)
        owner = NativeCallObservation(markers, "decode")
        owner.active = True
        owner.parallel = {**self.job["domain"]["parallel"], "rank": self.rank}
        owner.aliases = {m["id"]: m["id"] for members, _, _ in batches for m in members}
        cuts = FlowScopes(owner)
        markers.synchronize()
        with ScopedTasks.profile(self.output, self.rank), markers.window():
            for _, batch, _ in batches:
                DisaggregatedCalls(cuts).prepare(ScheduleBatch.prepare_for_prebuilt, batch)

        # The final fence and KV reads belong to qualification, outside the captured asynchronous calls.
        finished = torch.npu.Event()
        finished.record()
        finished.synchronize()
        environment = {**self.job["domain"], **markers.environment(), "parallel": owner.parallel}
        raw = owner.capture.document(environment)
        raw.update(markers.timing_evidence(), state_queries=cuts.queries)

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
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        started = monotonic_ns()
        torch.npu.set_device(0)
        pools = ReceiverPools(self.job)
        initialized = monotonic_ns()
        warmup, _ = pools.batch(self.job["batches"][0])
        ScheduleBatch.prepare_for_prebuilt(warmup)
        AscendMarkers.synchronize()
        pools.allocator.clear()
        pools.slots.clear()
        warmed = monotonic_ns()
        raw = self.observe(pools)

        exporting = monotonic_ns()
        sdk = self.output / "sdk"
        AscendExport.run(self.output, self.rank, sdk)
        raw["scopes"] = SGLangSourceScopes(raw["environment"]).lower(raw["scopes"])
        facts = AscendObservation(self.output, self.rank, raw, sdk / "trace.json", evidence=False).document()
        write_json(self.output / "facts.json", facts)
        write_json(self.output / "capture.json", {
            "framework_revision": revision, "measurements": "receiver.json", "facts": "facts.json",
            "engine_starts": 0, "model_runner_starts": 0,
            "initialization_ns": initialized - started, "warmup_ns": warmed - initialized,
            "capture_ns": exporting - warmed,
            "postprocessing_ns": monotonic_ns() - exporting,
        })
