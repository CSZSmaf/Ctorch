#include "dl/extension.h"

#include "core/layout.h"

namespace dl {

Index TensorSpec::numel() const { return detail::numel(shape); }
bool TensorSpec::is_contiguous() const { return detail::is_contiguous(shape, strides); }

TensorSpec tensor_spec(const Tensor& tensor) {
    return {tensor.shape(), tensor.strides(), tensor.dtype(), tensor.device(), tensor.storage_offset(),
        tensor.is_read_only()};
}

TensorSpec contiguous_spec(Shape shape, DType dtype, Device device) {
    auto strides = detail::contiguous_strides(shape);
    return {std::move(shape), std::move(strides), dtype, device, 0};
}

} // namespace dl
