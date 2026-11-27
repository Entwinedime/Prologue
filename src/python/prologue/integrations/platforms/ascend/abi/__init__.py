"""Framework ABI symbols backed by the Ascend device runtime."""

native_symbols = (
    ("sglang", "communicator_attribute", "npu_communicator"),
    ("sglang", "device_kv_manager", "sglang.srt.disaggregation.ascend.conn.AscendKVManager"),
    ("sglang", "mha_pool", "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMHATokenToKVPool"),
    ("sglang", "paged_allocator", "sglang.srt.hardware_backend.npu.allocator_npu.NPUPagedTokenToKVPoolAllocator"),
)
