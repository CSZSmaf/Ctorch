#include "backends/cuda_context.h"

#ifdef CTORCH_USE_CUDNN
#include "backends/cudnn.h"
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace dl::detail::cuda {
namespace {

template <class Release> void release_on_device(int device, Release release) noexcept {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    release();
    cudaSetDevice(previous);
}

cudaStream_t create_stream() {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}
cublasHandle_t create_blas() {
    cublasHandle_t handle;
    check(cublasCreate(&handle));
    return handle;
}
MemoryPool create_memory_pool(Device device) {
    cudaMemPoolProps properties{};
    properties.allocType = cudaMemAllocationTypePinned;
    properties.handleTypes = cudaMemHandleTypeNone;
    properties.location = {cudaMemLocationTypeDevice, device.index};
    cudaMemPool_t pool;
    check(cudaMemPoolCreate(&pool, &properties));
    MemoryPool result(pool, MemoryPoolDeleter{device.index});
    // 空闲页跨同步点保留，后续训练步继续从同一内存池分配。
    auto retention = std::numeric_limits<std::uint64_t>::max();
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &retention));
    return result;
}
void release_allocation(void* address, void* owner) noexcept {
    auto& context = *static_cast<Context*>(owner);
    release_on_device(context.device.index, [&] { cudaFreeAsync(address, context.stream.get()); });
}
void finish_external(void*, void* owner) noexcept {
    auto& context = *static_cast<Context*>(owner);
    release_on_device(context.device.index, [&] { cudaStreamSynchronize(context.stream.get()); });
}

} // namespace

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
    }
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS: ") + cublasGetStatusString(status));
    }
}

DeviceScope::DeviceScope(Device device) : selected_(device.index) {
    check(cudaGetDevice(&previous_));
    if (previous_ != selected_) {
        check(cudaSetDevice(selected_));
    }
}
DeviceScope::~DeviceScope() {
    if (previous_ != selected_) {
        cudaSetDevice(previous_);
    }
}
void StreamDeleter::operator()(cudaStream_t stream) const noexcept {
    release_on_device(device, [&] { cudaStreamDestroy(stream); });
}
void BlasDeleter::operator()(cublasHandle_t handle) const noexcept {
    release_on_device(device, [&] { cublasDestroy(handle); });
}
void MemoryPoolDeleter::operator()(cudaMemPool_t pool) const noexcept {
    release_on_device(device, [&] { cudaMemPoolDestroy(pool); });
}
void AsyncBufferDeleter::operator()(void* address) const noexcept {
    release_on_device(device, [&] { cudaFreeAsync(address, stream); });
}
void EventDeleter::operator()(cudaEvent_t event) const noexcept {
    release_on_device(device, [&] {
        if (wait) cudaEventSynchronize(event);
        cudaEventDestroy(event);
    });
}
Event make_event(Device device, bool wait_on_release) {
    DeviceScope selection(device);
    cudaEvent_t event;
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return Event(event, EventDeleter{device.index, wait_on_release});
}

Context::Context(Device device)
    : device(device), pool(create_memory_pool(device)), stream(create_stream(), StreamDeleter{device.index}),
      copy_stream(create_stream(), StreamDeleter{device.index}),
      blas(create_blas(), BlasDeleter{device.index}) {
    check(cublasSetStream(blas.get(), stream.get()));
    // GEMM 的 alpha/beta 作为宿主值提交，调用结束后局部标量即可离开作用域。
    check(cublasSetPointerMode(blas.get(), CUBLAS_POINTER_MODE_HOST));
    check(cublasSetMathMode(blas.get(), CUBLAS_DEFAULT_MATH));
#ifdef CTORCH_USE_CUDNN
    dnn = make_dnn_context(device, stream.get(), pool.get());
#endif
}

std::shared_ptr<Context> context(Device device) {
    static std::mutex mutex;
    static std::map<int, std::shared_ptr<Context>> contexts;
    const std::lock_guard lock(mutex);
    auto position = contexts.find(device.index);
    if (position != contexts.end()) {
        return position->second;
    }
    DeviceScope selection(device);
    auto result = std::make_shared<Context>(device);
    contexts.emplace(device.index, result);
    return result;
}

StreamHandoff::StreamHandoff(std::shared_ptr<Context> context, cudaStream_t consumer)
    : context_(std::move(context)), consumer_(consumer) {
    if (consumer_ == context_->stream.get()) return;
    DeviceScope selection(context_->device);
    // event 只传递 stream 依赖，其销毁由 CUDA 延后回收，无需阻塞宿主。
    event_ = make_event(context_->device, false);
    check(cudaEventRecord(event_.get(), context_->stream.get()));
    check(cudaStreamWaitEvent(consumer_, event_.get(), 0));
}

StreamHandoff::~StreamHandoff() {
    if (!event_) return;
    release_on_device(context_->device.index, [&] {
        // 最后一次消费已提交后记录事件，后续复用与 cudaFreeAsync 排在该依赖之后。
        cudaEventRecord(event_.get(), consumer_);
        cudaStreamWaitEvent(context_->stream.get(), event_.get(), 0);
    });
}

Allocation allocate(std::size_t bytes, Device device) {
    auto owner = context(device);
    DeviceScope selection(device);
    void* address;
    check(cudaMallocFromPoolAsync(&address, std::max(bytes, std::size_t{1}), owner->pool.get(), owner->stream.get()));
    return {address, release_allocation, std::move(owner)};
}
Allocation allocate_transfer(std::size_t bytes, Device device) {
    auto owner = context(device);
    DeviceScope selection(device);
    void* address;
    // 在复制 stream 上分配，输入传输可与已经提交的模型计算重叠。
    check(cudaMallocFromPoolAsync(&address, std::max(bytes, std::size_t{1}), owner->pool.get(), owner->copy_stream.get()));
    return {address, release_allocation, std::move(owner)};
}
Allocation allocate_pinned(std::size_t bytes) {
    void* address;
    check(cudaHostAlloc(&address, std::max(bytes, std::size_t{1}), cudaHostAllocPortable));
    return {address, [](void* pointer, void*) noexcept { cudaFreeHost(pointer); }, {}};
}
Allocation retain_external(void* address, std::shared_ptr<void> owner, Device device) {
    struct Lifetime {
        std::shared_ptr<Context> execution;
        std::shared_ptr<void> external;
    };
    auto lifetime = std::make_shared<Lifetime>(context(device), std::move(owner));
    // aliasing shared_ptr 同时保留外部对象与上下文，get() 仍指向 Execution 使用的 Context。
    auto execution = std::shared_ptr<void>(lifetime, lifetime->execution.get());
    return {address, finish_external, std::move(execution)};
}
void synchronize(Device device) {
    DeviceScope selection(device);
    const auto owner = context(device);
    check(cudaStreamSynchronize(owner->stream.get()));
    check(cudaStreamSynchronize(owner->copy_stream.get()));
}

void transfer(void* target, Device target_device, const void* source, Device source_device,
              std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (target_device.type == DeviceType::cuda && source_device.type == DeviceType::cuda
        && target_device != source_device) {
        // 跨 GPU 复制使用宿主暂存区，两个设备各自完成其 stream 上的传输。
        std::vector<std::byte> staging(bytes);
        transfer(staging.data(), Device::cpu(), source, source_device, bytes);
        transfer(target, target_device, staging.data(), Device::cpu(), bytes);
        return;
    }
    const auto device = target_device.type == DeviceType::cuda ? target_device : source_device;
    DeviceScope selection(device);
    auto owner = context(device);
    check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDefault, owner->stream.get()));
    // 显式复制返回时宿主缓冲已可读取或复用，设备算子本身继续异步排队。
    check(cudaStreamSynchronize(owner->stream.get()));
}

} // namespace dl::detail::cuda
