#include "dl/cuda.h"

#ifdef CTORCH_USE_CUDA
#include "backends/cuda_context.h"
#ifdef CTORCH_USE_CUDNN
#include "backends/cudnn.h"
#endif
#else
#include <stdexcept>
#endif

namespace dl::cuda {

#ifdef CTORCH_USE_CUDA
namespace {

// 上下文持有池，短作用域只切换设备；所有公共入口共用相同的资源寿命。
template <class Function>
decltype(auto) with_context(int device, Function function) {
    auto owner = detail::cuda::context(Device::cuda(device));
    detail::cuda::DeviceScope selection(owner->device);
    return function(*owner);
}

} // namespace

MemoryStats memory_stats(int device) {
    return with_context(device, [](auto& context) {
        auto read = [&](cudaMemPoolAttr attribute) {
            std::uint64_t value;
            detail::cuda::check(cudaMemPoolGetAttribute(context.pool.get(), attribute, &value));
            return value;
        };
        return MemoryStats{read(cudaMemPoolAttrUsedMemCurrent), read(cudaMemPoolAttrReservedMemCurrent),
            read(cudaMemPoolAttrUsedMemHigh), read(cudaMemPoolAttrReservedMemHigh),
            read(cudaMemPoolAttrReleaseThreshold)};
    });
}

void set_memory_release_threshold(std::uint64_t bytes, int device) {
    with_context(device, [&](auto& context) {
        detail::cuda::check(cudaMemPoolSetAttribute(context.pool.get(), cudaMemPoolAttrReleaseThreshold, &bytes));
    });
}

void reset_peak_memory_stats(int device) {
    with_context(device, [](auto& context) {
        std::uint64_t zero = 0;
        detail::cuda::check(cudaMemPoolSetAttribute(context.pool.get(), cudaMemPoolAttrUsedMemHigh, &zero));
        detail::cuda::check(cudaMemPoolSetAttribute(context.pool.get(), cudaMemPoolAttrReservedMemHigh, &zero));
    });
}

void release_unused_memory(int device) {
    with_context(device, [](auto& context) {
        detail::cuda::release_linear_workspace(*context.linear);
#ifdef CTORCH_USE_CUDNN
        detail::cuda::release_dnn_workspace(*context.dnn);
#endif
        detail::cuda::check(cudaStreamSynchronize(context.stream.get()));
        detail::cuda::check(cudaStreamSynchronize(context.copy_stream.get()));
        detail::cuda::check(cudaMemPoolTrimTo(context.pool.get(), 0));
    });
}

#else
namespace {
[[noreturn]] void unavailable() { throw std::runtime_error("Ctorch was built without CUDA support"); }
}
MemoryStats memory_stats(int) { unavailable(); }
void set_memory_release_threshold(std::uint64_t, int) { unavailable(); }
void reset_peak_memory_stats(int) { unavailable(); }
void release_unused_memory(int) { unavailable(); }
#endif

} // namespace dl::cuda
