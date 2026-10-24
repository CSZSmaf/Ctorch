#include "dl/transfer.h"

#include "autograd/record.h"
#include "core/tensor_impl.h"
#include "ops/numeric.h"

#ifdef CTORCH_USE_CUDA
#include "backends/cuda_context.h"
#endif

#include <stdexcept>

namespace dl {
namespace {

#ifdef CTORCH_USE_CUDA
detail::TensorData pinned_data(Shape shape, DType dtype) {
    const auto bytes = static_cast<std::size_t>(detail::numel(shape)) * element_size(dtype);
    auto storage = std::make_shared<detail::Storage>(detail::cuda::allocate_pinned(bytes), bytes, Device::cpu(), true);
    return {std::move(storage), detail::contiguous_layout(std::move(shape)), dtype};
}
Tensor transferred(detail::TensorData data, const Tensor& source) {
    return detail::record(std::move(data), {detail::gradient_edge(source)},
        [device = source.device()](const auto& grad, auto& sink, const auto& edges) {
            sink.add(edges[0], detail::numeric::to(grad, device));
        });
}
#endif

} // namespace

bool Tensor::is_pinned() const { return detail::data_of(*this).storage->pinned; }

Tensor Tensor::pin_memory() const {
#ifdef CTORCH_USE_CUDA
    if (is_pinned() && is_contiguous()) return *this;
    auto result = pinned_data(shape(), dtype());
    detail::numeric::copy_into(result, detail::data_of(*this));
    return transferred(std::move(result), *this);
#else
    throw std::runtime_error("pinned memory requires a CUDA build");
#endif
}

struct Transfer::Impl {
    explicit Impl(const Tensor& input, Device target) : output(input) {
        if (input.device() == target) return;
#ifdef CTORCH_USE_CUDA
        if (input.device().type == DeviceType::cpu || target.type == DeviceType::cpu) {
            const auto device = target.type == DeviceType::cuda ? target : input.device();
            owner = detail::cuda::context(device);
            detail::cuda::DeviceScope selection(device);
            const auto bytes = static_cast<std::size_t>(input.numel()) * element_size(input.dtype());
            const auto stream = owner->copy_stream.get();
            completed = detail::cuda::make_event(device);
            if (target.type == DeviceType::cuda) {
                source = input.detach().pin_memory();
                auto storage = std::make_shared<detail::Storage>(detail::cuda::allocate_transfer(bytes, device), bytes, device);
                detail::TensorData data{std::move(storage), detail::contiguous_layout(input.shape()), input.dtype()};
                detail::cuda::check(cudaMemcpyAsync(data.mutable_address(), detail::data_of(*source).address(),
                    bytes, cudaMemcpyHostToDevice, stream));
                output = transferred(std::move(data), input);
            } else {
                source = input.detach().contiguous();
                prepared = detail::cuda::make_event(device);
                detail::cuda::check(cudaEventRecord(prepared.get(), owner->stream.get()));
                detail::cuda::check(cudaStreamWaitEvent(stream, prepared.get(), 0));
                auto data = pinned_data(input.shape(), input.dtype());
                detail::cuda::check(cudaMemcpyAsync(data.mutable_address(), detail::data_of(*source).address(),
                    bytes, cudaMemcpyDeviceToHost, stream));
                output = transferred(std::move(data), input);
            }
            detail::cuda::check(cudaEventRecord(completed.get(), stream));
            return;
        }
#endif
        output = input.to(target);
    }

    Tensor output;
    std::optional<Tensor> source;
#ifdef CTORCH_USE_CUDA
    std::shared_ptr<detail::cuda::Context> owner;
    // 逆序销毁先等待完成事件，再释放源缓冲与输出。
    detail::cuda::Event prepared, completed;
#endif
};

Transfer::Transfer(const Tensor& input, Device target) : impl_(std::make_unique<Impl>(input, target)) {}
Transfer::~Transfer() = default;
Transfer::Transfer(Transfer&&) noexcept = default;
Transfer& Transfer::operator=(Transfer&&) noexcept = default;

void Transfer::wait() const {
#ifdef CTORCH_USE_CUDA
    if (impl_->completed) {
        detail::cuda::DeviceScope selection(impl_->owner->device);
        detail::cuda::check(cudaEventSynchronize(impl_->completed.get()));
    }
#endif
}

bool Transfer::ready() const {
#ifdef CTORCH_USE_CUDA
    if (impl_->completed) {
        detail::cuda::DeviceScope selection(impl_->owner->device);
        const auto result = cudaEventQuery(impl_->completed.get());
        if (result == cudaErrorNotReady) return false;
        detail::cuda::check(result);
    }
#endif
    return true;
}

Tensor Transfer::tensor() const {
#ifdef CTORCH_USE_CUDA
    if (impl_->completed && impl_->output.device().type == DeviceType::cuda) {
        detail::cuda::DeviceScope selection(impl_->owner->device);
        detail::cuda::check(cudaStreamWaitEvent(impl_->owner->stream.get(), impl_->completed.get(), 0));
        return impl_->output;
    }
#endif
    wait();
    return impl_->output;
}

} // namespace dl
