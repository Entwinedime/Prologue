#include <ATen/core/Tensor.h>
#include <acl/acl_rt.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <utility>

namespace {

void move_weight(const at::Tensor & weight, const at::Tensor & backing, bool to_device) {
    const auto storage = weight.storage();
    const auto bytes = storage.nbytes();
    TORCH_CHECK(weight.device().is_privateuseone() && backing.device().is_cpu() && backing.is_contiguous()
                    && backing.scalar_type() == at::kByte && backing.nbytes() == bytes && bytes > 0,
                "weight parking needs an Ascend storage and an equally sized contiguous host byte buffer");

    // Keeping the StorageImpl preserves the Ascend format descriptor, strides and every native weight view.
    if (to_device) {
        TORCH_CHECK(!storage.data(), "weight storage is already resident");
        auto allocation = storage.allocator()->allocate(bytes);
        const auto status = aclrtMemcpy(allocation.get(), bytes, backing.const_data_ptr(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(status == ACL_SUCCESS, "restoring Ascend weight storage failed with ", status);
        storage.set_data_ptr(std::move(allocation));
    } else {
        TORCH_CHECK(storage.data(), "weight storage is already parked");
        const auto status = aclrtMemcpy(backing.mutable_data_ptr(), bytes, storage.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        TORCH_CHECK(status == ACL_SUCCESS, "parking Ascend weight storage failed with ", status);
        storage.set_data_ptr(c10::DataPtr(nullptr, weight.device()));
    }
}

} // namespace

TORCH_LIBRARY(prologue_ascend, library) {
    library.def("move_weight_(Tensor(a!) weight, Tensor(b!) backing, bool to_device) -> ()", &move_weight);
}
