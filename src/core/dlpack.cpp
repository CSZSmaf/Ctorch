#include "dl/dlpack.h"

#include "core/tensor_impl.h"
#ifdef CTORCH_USE_CUDA
#include "backends/cuda_context.h"
#endif

#include <algorithm>
#include <array>
#include <stdexcept>

namespace dl {
namespace {

// 顺序与 DType 一致，每个标量使用一个 lane，bool 与框架的单字节表示对应。
constexpr std::array<DLDataType, 6> element_types{{
    {kDLFloat, 32, 1}, {kDLFloat, 64, 1}, {kDLInt, 64, 1},
    {kDLBool, 8, 1}, {kDLFloat, 16, 1}, {kDLBfloat, 16, 1}
}};

DType dtype_of_dlpack(DLDataType value) {
    const auto position = std::find_if(element_types.begin(), element_types.end(), [=](DLDataType type) {
        return type.code == value.code && type.bits == value.bits && type.lanes == value.lanes;
    });
    if (position == element_types.end()) throw std::runtime_error("unsupported DLPack element type");
    return static_cast<DType>(position - element_types.begin());
}

Device device_of_dlpack(DLDevice device) {
    switch (device.device_type) {
    case kDLCPU:
    case kDLCUDAHost: return Device::cpu();
    case kDLCUDA: return Device::cuda(device.device_id);
    default: throw std::runtime_error("unsupported DLPack device type");
    }
}

DLDevice dlpack_device(const detail::TensorData& data) {
    if (data.device().type == DeviceType::cuda) return {kDLCUDA, data.device().index};
    return {data.storage->pinned ? kDLCUDAHost : kDLCPU, 0};
}

template <class Managed> struct Exported {
    explicit Exported(const detail::TensorData& source, std::intptr_t consumer_stream) : data(source) {
        const bool empty = data.layout.numel() == 0;
        managed.dl_tensor = {empty ? nullptr : data.storage->allocation.address, dlpack_device(data),
            static_cast<std::int32_t>(data.layout.sizes.size()), element_types[static_cast<std::size_t>(data.dtype)],
            data.layout.sizes.data(), data.layout.strides.data(),
            empty ? 0 : static_cast<std::uint64_t>(data.layout.offset) * element_size(data.dtype)};
        managed.manager_ctx = this;
        managed.deleter = [](Managed* value) { delete static_cast<Exported*>(value->manager_ctx); };
#ifdef CTORCH_USE_CUDA
        if (data.device().type == DeviceType::cuda && consumer_stream != -1) {
            auto context = std::static_pointer_cast<detail::cuda::Context>(data.storage->allocation.context);
            const auto stream = consumer_stream == 1 ? cudaStreamLegacy
                : consumer_stream == 2 ? cudaStreamPerThread : reinterpret_cast<cudaStream_t>(consumer_stream);
            handoff = std::make_unique<detail::cuda::StreamHandoff>(std::move(context), stream);
        }
#else
        (void)consumer_stream;
#endif
    }

    // 元数据随描述符独立保存；逆序销毁先交回消费进度，再释放共享 Storage。
    detail::TensorData data;
#ifdef CTORCH_USE_CUDA
    std::unique_ptr<detail::cuda::StreamHandoff> handoff;
#endif
    Managed managed{};
};

template <class Managed> std::unique_ptr<Managed, DLPackDeleter> export_tensor(
    const Tensor& tensor, std::intptr_t consumer_stream) {
    auto owner = std::make_unique<Exported<Managed>>(detail::data_of(tensor), consumer_stream);
    auto* managed = &owner->managed;
    owner.release();
    return std::unique_ptr<Managed, DLPackDeleter>(managed);
}

template <class Managed> Tensor import_tensor(std::unique_ptr<Managed, DLPackDeleter> tensor, bool read_only) {
    const auto& source = tensor->dl_tensor;
    const auto dtype = dtype_of_dlpack(source.dtype);
    const auto device = device_of_dlpack(source.device);
    Shape shape = source.ndim == 0 ? Shape{} : Shape(source.shape, source.shape + source.ndim);
    auto strides = source.ndim == 0 || !source.strides
        ? detail::contiguous_strides(shape) : Shape(source.strides, source.strides + source.ndim);
    if (std::any_of(strides.begin(), strides.end(), [](Index stride) { return stride < 0; }))
        throw std::runtime_error("DLPack import requires nonnegative strides");
    detail::Layout layout{std::move(shape), std::move(strides)};

    // byte_offset 折入这次借用的存储起点，后续视图继续使用元素单位的相对偏移。
    // span 包含布局间隙；零元素张量保留空存储，不对空地址做偏移运算。
    void* address = source.data;
    Index span = 0;
    if (layout.numel() != 0) {
        address = static_cast<std::byte*>(source.data) + source.byte_offset;
        span = 1;
        for (std::size_t i = 0; i < layout.sizes.size(); ++i)
            span += (layout.sizes[i] - 1) * layout.strides[i];
    }
    const bool pinned = source.device.device_type == kDLCUDAHost;
    std::shared_ptr<void> owner(std::move(tensor));
    detail::Allocation allocation{address, [](void*, void*) noexcept {}, std::move(owner)};
    if (device.type == DeviceType::cuda) {
#ifdef CTORCH_USE_CUDA
        allocation = detail::cuda::retain_external(address, std::move(allocation.context), device);
#else
        throw std::runtime_error("Ctorch was built without CUDA support");
#endif
    }
    auto storage = std::make_shared<detail::Storage>(std::move(allocation),
        static_cast<std::size_t>(span) * element_size(dtype), device, pinned, read_only);
    return detail::make_tensor({std::move(storage), std::move(layout), dtype});
}

} // namespace

void DLPackDeleter::operator()(DLManagedTensorVersioned* tensor) const noexcept {
    if (tensor && tensor->deleter) tensor->deleter(tensor);
}
void DLPackDeleter::operator()(DLManagedTensor* tensor) const noexcept {
    if (tensor && tensor->deleter) tensor->deleter(tensor);
}

std::intptr_t dlpack_consumer_stream(Device device) {
    if (device.type == DeviceType::cpu) return 0;
#ifdef CTORCH_USE_CUDA
    return reinterpret_cast<std::intptr_t>(detail::cuda::context(device)->stream.get());
#else
    throw std::runtime_error("Ctorch was built without CUDA support");
#endif
}

DLPackTensor to_dlpack(const Tensor& tensor, std::intptr_t consumer_stream) {
    auto result = export_tensor<DLManagedTensorVersioned>(tensor, consumer_stream);
    result->version = {1, 0};
    result->flags = tensor.is_read_only() ? DLPACK_FLAG_BITMASK_READ_ONLY : 0;
    return result;
}

LegacyDLPackTensor to_dlpack_legacy(const Tensor& tensor, std::intptr_t consumer_stream) {
    if (tensor.is_read_only())
        throw std::runtime_error("legacy DLPack cannot describe read-only storage; clone before exporting");
    return export_tensor<DLManagedTensor>(tensor, consumer_stream);
}

Tensor from_dlpack(DLPackTensor tensor) {
    // major 不一致时仅访问版本和 deleter；未知布局字段留给生产者自己的释放函数。
    if (tensor->version.major != 1) throw std::runtime_error("unsupported DLPack ABI major version");
    constexpr std::uint64_t flags = DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED;
    if (tensor->flags & ~flags) throw std::runtime_error("unsupported DLPack tensor flags");
    const bool read_only = (tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
    return import_tensor(std::move(tensor), read_only);
}

Tensor from_dlpack(LegacyDLPackTensor tensor) { return import_tensor(std::move(tensor), false); }

} // namespace dl
