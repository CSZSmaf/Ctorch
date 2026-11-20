#include "dl/tensor.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <cstring>
#include <stdexcept>

namespace dl {

const Shape& Tensor::shape() const { return impl_->data.layout.sizes; }
const Shape& Tensor::strides() const { return impl_->data.layout.strides; }
Index Tensor::size(Index axis) const { return shape()[detail::axis_index(axis, ndim())]; }
Index Tensor::ndim() const { return static_cast<Index>(shape().size()); }
Index Tensor::numel() const { return impl_->data.layout.numel(); }
Index Tensor::storage_offset() const { return impl_->data.layout.offset; }
DType Tensor::dtype() const { return impl_->data.dtype; }
Device Tensor::device() const { return impl_->data.storage->device; }
bool Tensor::is_contiguous() const { return impl_->data.layout.is_contiguous(); }
bool Tensor::is_read_only() const { return impl_->data.storage->read_only; }
bool Tensor::requires_grad() const { return bool(impl_->autograd); }
bool Tensor::is_leaf() const { return !impl_->autograd || bool(impl_->autograd->leaf); }
const void* Tensor::raw_data() const { return impl_->data.address(); }

void Tensor::requires_grad_(bool enabled) const {
    if (enabled == requires_grad()) return;
    impl_->autograd = enabled ? detail::make_leaf_meta() : nullptr;
}

Tensor Tensor::detach() const { return detail::make_tensor(impl_->data); }

void Tensor::backward() const {
    if (!impl_->autograd) throw std::logic_error("backward requires a tensor that tracks gradients");
    detail::run_backward(impl_->autograd->edge, detail::backward_seed(*this));
}

void Tensor::backward(const Tensor& seed) const {
    if (!impl_->autograd) throw std::logic_error("backward requires a tensor that tracks gradients");
    detail::run_backward(impl_->autograd->edge, detail::backward_seed(*this, &seed));
}

std::optional<Tensor> Tensor::grad() const {
    if (impl_->autograd && impl_->autograd->leaf && impl_->autograd->leaf->value) {
        return detail::make_tensor(*impl_->autograd->leaf->value);
    }
    return std::nullopt;
}

void Tensor::zero_grad() const {
    if (impl_->autograd && impl_->autograd->leaf) {
        impl_->autograd->leaf->value.reset();
    }
}

Tensor full(Shape shape, double value, TensorOptions options) {
    return detail::make_tensor(detail::numeric::full(std::move(shape), value, options.dtype, options.device),
                               options.requires_grad);
}
Tensor zeros(Shape shape, TensorOptions options) { return full(std::move(shape), 0.0, options); }
Tensor ones(Shape shape, TensorOptions options) { return full(std::move(shape), 1.0, options); }

namespace detail {

TensorData backward_seed(const Tensor& output, const Tensor* seed) {
    const auto dtype = accumulation_dtype(output.dtype());
    if (!seed) {
        if (output.numel() != 1) throw std::invalid_argument("non-scalar output requires an explicit gradient seed");
        return numeric::full(output.shape(), 1.0, dtype, output.device());
    }
    if (seed->shape() != output.shape() || seed->device() != output.device())
        throw std::invalid_argument("gradient seed must match the output shape and device");
    if (seed->dtype() != output.dtype() && seed->dtype() != dtype)
        throw std::invalid_argument("gradient seed must use the output or accumulation dtype");
    return numeric::cast(data_of(*seed), dtype);
}

Tensor from_host(const void* source, Shape shape, DType dtype, bool requires_grad, Device device) {
    auto data = empty_data(std::move(shape), dtype);
    const auto bytes = data.storage->bytes;
    if (bytes != 0) {
        std::memcpy(data.mutable_address(), source, bytes);
    }
    return make_tensor(numeric::to(data, device), requires_grad);
}

} // namespace detail
} // namespace dl
