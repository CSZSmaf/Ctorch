#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

#include <algorithm>
#include <stdexcept>

namespace dl::detail::numeric {

TensorData reshape(TensorData input, Shape shape) {
    shape = resolve_shape(std::move(shape), input.layout.numel());
    auto strides = reshape_strides(input.layout, shape);
    if (!strides) input = clone(input);
    auto steps = strides ? std::move(*strides) : contiguous_strides(shape);
    input.layout = {std::move(shape), std::move(steps), input.layout.offset};
    return input;
}

TensorData transpose(const TensorData& input, Index first, Index second) {
    auto result = input;
    std::swap(result.layout.sizes[first], result.layout.sizes[second]);
    std::swap(result.layout.strides[first], result.layout.strides[second]);
    return result;
}

TensorData permute(const TensorData& input, const Shape& axes) {
    auto result = input;
    for (std::size_t dim = 0; dim < axes.size(); ++dim) {
        result.layout.sizes[dim] = input.layout.sizes[axes[dim]];
        result.layout.strides[dim] = input.layout.strides[axes[dim]];
    }
    return result;
}

TensorData expand(const TensorData& input, Shape shape) {
    auto result = input;
    auto strides = broadcast_strides(input.layout, shape);
    result.layout = {std::move(shape), std::move(strides), input.layout.offset};
    return result;
}

TensorData slice(const TensorData& input, Index axis, Index start, Index stop, Index step) {
    auto result = input;
    result.layout.offset += start * result.layout.strides[axis];
    result.layout.sizes[axis] = (stop - start + step - 1) / step;
    result.layout.strides[axis] *= step;
    return result;
}

void slice_backward_into(TensorData& target, const TensorData& grad,
                         Index axis, Index start, Index step, bool accumulate) {
    auto selected = slice(target, axis, start, start + grad.layout.sizes[axis] * step, step);
    if (accumulate) add_into(selected, grad);
    else copy_into(selected, grad);
}

} // namespace dl::detail::numeric

namespace dl {
namespace {

Tensor shape_result(detail::TensorData result, const Tensor& input) {
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [shape = input.shape()](auto grad, auto& sink, const auto& edges) {
            sink.add_transferred(edges[0], detail::numeric::reshape(std::move(grad), shape));
        });
}

} // namespace

Tensor flatten(const Tensor& input, Index first, Index last) {
    if (input.ndim() == 0) {
        return input.reshape({1});
    }
    first = detail::axis_index(first, input.ndim());
    last = detail::axis_index(last, input.ndim());
    if (first == last) {
        return input;
    }
    auto shape = input.shape();
    shape[first] = detail::numel(std::span(shape).subspan(first, last - first + 1));
    shape.erase(shape.begin() + first + 1, shape.begin() + last + 1);
    return input.reshape(std::move(shape));
}

Tensor Tensor::clone() const {
    return detail::record(detail::numeric::clone(detail::data_of(*this)),
        {detail::gradient_edge(*this)}, [](auto grad, auto& sink, const auto& edges) {
            sink.add_transferred(edges[0], std::move(grad));
        });
}

Tensor Tensor::contiguous() const { return is_contiguous() ? *this : clone(); }

Tensor Tensor::to(Device device) const {
    if (device == this->device()) {
        return *this;
    }
    return detail::record(detail::numeric::to(detail::data_of(*this), device),
        {detail::gradient_edge(*this)},
        [source = this->device()](const auto& grad, auto& sink, const auto& edges) {
            sink.add_owned(edges[0], detail::numeric::to(grad, source));
        });
}

Tensor Tensor::astype(DType dtype) const {
    if (dtype == this->dtype()) {
        return *this;
    }
    auto result = detail::numeric::cast(detail::data_of(*this), dtype);
    if (!is_floating(dtype)) {
        return detail::make_tensor(std::move(result));
    }
    return detail::record(std::move(result), {detail::gradient_edge(*this)},
        [source_type = this->dtype()](auto grad, auto& sink, const auto& edges) {
            grad = detail::numeric::cast(grad, accumulation_dtype(source_type));
            sink.add_transferred(edges[0], std::move(grad));
        });
}

Tensor Tensor::view(Shape shape) const {
    shape = detail::resolve_shape(std::move(shape), numel());
    auto data = detail::data_of(*this);
    auto strides = detail::reshape_strides(data.layout, shape);
    if (!strides) throw std::invalid_argument("view cannot represent this layout; use reshape to copy the logical values");
    data.layout = {std::move(shape), std::move(*strides), data.layout.offset};
    return shape_result(std::move(data), *this);
}

Tensor Tensor::reshape(Shape shape) const {
    return shape_result(detail::numeric::reshape(detail::data_of(*this), std::move(shape)), *this);
}

Tensor Tensor::flatten(Index first, Index last) const { return dl::flatten(*this, first, last); }

Tensor Tensor::unsqueeze(Index axis) const {
    axis = detail::axis_index(axis, ndim() + 1);
    auto data = detail::data_of(*this);
    const Index step = axis == ndim() ? 1 : data.layout.strides[axis] * data.layout.sizes[axis];
    data.layout.sizes = detail::insert_axis(data.layout.sizes, axis, 1);
    data.layout.strides = detail::insert_axis(data.layout.strides, axis, step);
    return shape_result(std::move(data), *this);
}

Tensor Tensor::squeeze() const {
    auto data = detail::data_of(*this);
    std::size_t target = 0;
    for (std::size_t axis = 0; axis < data.layout.sizes.size(); ++axis) {
        if (data.layout.sizes[axis] != 1) {
            data.layout.sizes[target] = data.layout.sizes[axis];
            data.layout.strides[target++] = data.layout.strides[axis];
        }
    }
    data.layout.sizes.resize(target);
    data.layout.strides.resize(target);
    return shape_result(std::move(data), *this);
}

Tensor Tensor::squeeze(Index axis) const {
    if (ndim() == 0) return *this;
    axis = detail::axis_index(axis, ndim());
    if (size(axis) != 1) return *this;
    auto data = detail::data_of(*this);
    data.layout.sizes.erase(data.layout.sizes.begin() + axis);
    data.layout.strides.erase(data.layout.strides.begin() + axis);
    return shape_result(std::move(data), *this);
}

Tensor Tensor::select(Index axis, Index index) const {
    if (index < 0) index += size(axis);
    return slice(axis, index, index + 1).squeeze(axis);
}

Tensor Tensor::narrow(Index axis, Index start, Index length) const {
    if (start < 0) start += size(axis);
    return slice(axis, start, start + length);
}

Tensor Tensor::transpose(Index first, Index second) const {
    first = detail::axis_index(first, ndim());
    second = detail::axis_index(second, ndim());
    return detail::record(detail::numeric::transpose(detail::data_of(*this), first, second),
        {detail::gradient_edge(*this)},
        [first, second](const auto& grad, auto& sink, const auto& edges) {
            sink.add(edges[0], detail::numeric::transpose(grad, first, second));
        });
}

Tensor Tensor::permute(Shape axes) const {
    Shape inverse(axes.size());
    for (std::size_t dim = 0; dim < axes.size(); ++dim) {
        axes[dim] = detail::axis_index(axes[dim], ndim());
        inverse[axes[dim]] = static_cast<Index>(dim);
    }
    return detail::record(detail::numeric::permute(detail::data_of(*this), axes),
        {detail::gradient_edge(*this)},
        [inverse = std::move(inverse)](const auto& grad, auto& sink, const auto& edges) {
            sink.add(edges[0], detail::numeric::permute(grad, inverse));
        });
}

Tensor Tensor::slice(Index axis, Index start, Index stop, Index step) const {
    axis = detail::axis_index(axis, ndim());
    return detail::record(detail::numeric::slice(detail::data_of(*this), axis, start, stop, step),
        {detail::gradient_edge(*this)},
        [shape = shape(), axis, start, step](const auto& grad, auto& sink, const auto& edges) {
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::slice_backward_into(target.value, grad, axis, start, step, target.accumulate);
        });
}

Tensor Tensor::expand(Shape shape) const {
    return detail::record(detail::numeric::expand(detail::data_of(*this), std::move(shape)),
        {detail::gradient_edge(*this)},
        [original = this->shape()](const auto& grad, auto& sink, const auto& edges) {
            sink.add_reduced(edges[0], grad, original);
        });
}

} // namespace dl
