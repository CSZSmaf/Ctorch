#pragma once

#include "core/tensor_data.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <type_traits>

namespace dl::detail::cuda {

void check(cudaError_t status);
void check(cublasStatus_t status);

/// 一次宿主调用临时选择目标设备，返回时恢复调用者的设备选择。
class DeviceScope {
public:
    explicit DeviceScope(Device device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_;
    int selected_;
};

struct StreamDeleter {
    int device;
    void operator()(cudaStream_t stream) const noexcept;
};
struct BlasDeleter {
    int device;
    void operator()(cublasHandle_t handle) const noexcept;
};
struct MemoryPoolDeleter {
    int device;
    void operator()(cudaMemPool_t pool) const noexcept;
};
struct AsyncBufferDeleter {
    int device;
    cudaStream_t stream;
    void operator()(void* address) const noexcept;
};
using MemoryPool = std::unique_ptr<std::remove_pointer_t<cudaMemPool_t>, MemoryPoolDeleter>;
struct EventDeleter {
    int device = 0;
    bool wait = true;
    void operator()(cudaEvent_t event) const noexcept;
};
using Event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, EventDeleter>;
Event make_event(Device device, bool wait_on_release = true);

struct DnnContext;

/// 每设备拥有独立的 Runtime 内存池与计算、复制两个 stream，数值库绑定计算 stream。
/// Storage 持有上下文；池最后销毁，使工作区和张量的有序归还先结束其宿主生命周期。
struct Context {
    explicit Context(Device device);
    Device device;
    MemoryPool pool;
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter> stream;
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter> copy_stream;
    std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, BlasDeleter> blas;
    std::shared_ptr<DnnContext> dnn;
};

std::shared_ptr<Context> context(Device device);

/// 外部 stream 先等待本框架的结果，作用域结束时将消费进度交回计算 stream。
/// 外部 stream 及其提交线程约定由调用方保持到本对象销毁。
class StreamHandoff {
public:
    StreamHandoff(std::shared_ptr<Context> context, cudaStream_t consumer);
    ~StreamHandoff();
    StreamHandoff(const StreamHandoff&) = delete;
    StreamHandoff& operator=(const StreamHandoff&) = delete;

private:
    std::shared_ptr<Context> context_;
    cudaStream_t consumer_;
    Event event_;
};

struct Execution {
    explicit Execution(const TensorData& data)
        : selection(data.device()), context(*static_cast<Context*>(data.storage->allocation.context.get())) {}

    DeviceScope selection;
    Context& context;
    cudaStream_t stream() const { return context.stream.get(); }
};

Allocation allocate(std::size_t bytes, Device device);
Allocation allocate_transfer(std::size_t bytes, Device device);
Allocation allocate_pinned(std::size_t bytes);
/// 保留外部所有者，同时使 Execution 继续取得本设备的计算上下文。
/// 最后一个 Storage 释放时等待计算结束，再归还外部资源。
Allocation retain_external(void* address, std::shared_ptr<void> owner, Device device);
void synchronize(Device device);
void transfer(void* target, Device target_device, const void* source, Device source_device,
              std::size_t bytes);

} // namespace dl::detail::cuda
