"""Prepare native KV in one SGLang ModelRunner and time its warmed Decode attention."""

from time import monotonic_ns

from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.ascend.attention_observation import DecodeAttentionObservation
from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration
from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks


class AttentionReturned(Exception):
    """The requested native layer has committed its output; later model layers are outside the cut."""


class NativeAttentionBatch:
    def __init__(self, native, runner):
        self.native, self.runner = native, runner
        self.model = runner.torch_runner
        self.cache = native.TreeCacheNamespace(page_size=self.model.server_args.page_size, device=self.model.device,
                                              token_to_kv_pool_allocator=self.model.token_to_kv_pool_allocator)

    @staticmethod
    def stop_after_call(module, args, result):
        raise AttentionReturned

    def forward(self, batch):
        import torch

        self.native._maybe_prepare_mlp_sync_batch(batch, self.model)
        forward = self.native.ForwardBatch.init_new(batch, self.model)
        try:
            with torch.no_grad():
                self.model.forward(forward)
        except AttentionReturned:
            self.runner.synchronize()
            return
        raise ValueError("the native forward did not reach the requested attention site")

    def extend(self, lengths):
        self.runner.clear()
        vocab = self.model.model_config.vocab_size
        ids = [[(index + pos) % vocab for pos in range(n)] for index, n in enumerate(lengths)]
        requests = self.native.prepare_synthetic_inputs_for_latency_test(len(ids), max(lengths), ids)
        batch = self.native.ScheduleBatch.init_new(reqs=requests, req_to_token_pool=self.model.req_to_token_pool,
                                                  token_to_kv_pool_allocator=self.model.token_to_kv_pool_allocator,
                                                  tree_cache=self.cache, model_config=self.model.model_config,
                                                  enable_overlap=False, spec_algorithm=self.native.SpeculativeAlgorithm.NONE)
        batch.prepare_for_extend()
        if batch.input_ids is None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None
        self.forward(batch)
        return batch

    def decode(self, batch):
        import torch

        # The local dense work query fixes geometry. Feed a declared token without running an unrelated logits head.
        tokens = [req.origin_input_ids[-1] for req in batch.reqs]
        batch.input_ids = torch.tensor(tokens, dtype=torch.int64, device=self.model.device)
        batch.prepare_for_decode()
        self.forward(batch)


class KvReadback:
    def __init__(self, pool, module):
        self.pool = pool
        self.layer = module.layer_id - pool.start_layer
        self.operands = None

    def retain(self, module, args):
        self.operands = args[1:3]

    def read(self, batch, locations, extend):
        import torch

        slots = batch.req_pool_indices_cpu.tolist()
        lengths = batch.seq_lens_cpu.tolist()
        table = [batch.req_to_token_pool.req_to_token[slot, :length].tolist() for slot, length in zip(slots, lengths, strict=True)]
        written = batch.out_cache_loc.tolist()
        flat = [loc for row in table for loc in row]
        expected = flat if extend else [row[-1] for row in table]
        correct = (written == expected and len(set(flat)) == len(flat) and min(flat) > 0
                   and lengths == [request.kv_committed_len for request in batch.reqs])
        matching_prefix = extend or [row[:-1] for row in table] == locations

        # Read just the requested slots after the device interval; preserve both keys and values for the next read.
        pool = self.pool
        contents = []
        for buffer, operand in zip((pool.k_buffer, pool.v_buffer), self.operands, strict=True):
            tokens = buffer[self.layer].flatten(0, 1)
            now = tokens[flat].cpu()
            expected_data = operand.reshape(-1, *tokens.shape[1:]).cpu()
            if extend:
                correct = correct and torch.equal(now, expected_data)
            else:
                pieces = now.split(lengths)
                correct = correct and torch.equal(torch.stack([piece[-1] for piece in pieces]), expected_data)
                matching_prefix = matching_prefix and torch.equal(torch.cat([p[:-1] for p in pieces]), self.prefix[len(contents)])
            contents.append(now)
        self.operands = None
        if extend:
            self.prefix = contents
        return {"slots": slots, "locations": table, "written": written, "committed": lengths,
                "matching_operands": correct, "preserved_prefix": matching_prefix}


class AttentionRunner:
    def __init__(self, job, output):
        self.job, self.output = job, output

    def measure(self, lengths, observer, kv):
        started = monotonic_ns()
        batch = self.batch.extend(lengths)
        prefill = kv.read(batch, None, True)
        prepared = monotonic_ns()
        self.batch.decode(batch)
        decoded = monotonic_ns()
        read = kv.read(batch, prefill["locations"], False)
        call = observer.rows[-1]
        call["kv"] = {"prefill": prefill, "decode": read}
        return {"context": lengths, "prepare_and_read_ns": prepared - started,
                "decode_ns": decoded - prepared, "readback_ns": monotonic_ns() - decoded}

    def run(self, revision):
        started = monotonic_ns()
        from sglang.benchmark import one_batch

        config = NativeConfiguration(self.job["runner"]["configuration"])
        args = one_batch.ServerArgs(**config.runner_options(self.job["runner"]["model"]))
        one_batch._set_envs_and_config(args)
        one_batch.initialize_moe_config(args)
        one_batch.initialize_fp8_gemm_config(args)
        one_batch.initialize_fp4_gemm_config(args)
        runner, _ = one_batch.load_model(args, one_batch.PortArgs.init_new(args), gpu_id=0, tp_rank=0)
        initialized = monotonic_ns()
        try:
            actual = runner.torch_runner
            self.batch = NativeAttentionBatch(one_batch, runner)
            site = self.job["site"]
            module = actual.model.get_submodule(site)
            with module.register_forward_hook(self.batch.stop_after_call):
                for lengths in self.job["contexts"]:
                    self.batch.decode(self.batch.extend(lengths))
                warmed = monotonic_ns()
                observer = DecodeAttentionObservation(None, {"output": self.output, "rank_offset": 0, "attention_sites": {"0": [site]}})
                observer.attach(actual)
                kv = KvReadback(actual.token_to_kv_pool, module)
                timings = []
                with module.register_forward_pre_hook(kv.retain):
                    for lengths in self.job["contexts"]:
                        for _ in range(2):
                            timings.append(self.measure(lengths, observer, kv))
            record = observer.document()
            record.update(model=actual.model_config.hf_config.to_dict(), rank=observer.rank, load_format=args.load_format)
            runner.clear()
        finally:
            one_batch.destroy_model_parallel()
            one_batch.destroy_distributed_environment()

        # SDK postprocessing starts after the measured worker has released its process group.
        exporting = monotonic_ns()
        ScopedTasks.export_calls(self.output, observer.rank, record["calls"])
        write_json(self.output / "attention.json", record)
        write_json(self.output / "capture.json", {
            "framework_revision": revision, "measurements": "attention.json", "engine_starts": 0, "model_runner_starts": 1,
            "initialization_ns": initialized - started, "warmup_ns": warmed - initialized,
            "points": timings, "postprocessing_ns": monotonic_ns() - exporting,
        })
