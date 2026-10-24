#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl {

std::vector<Tensor> Tensor::split(Index split_size, Index axis) const {
    const auto length = size(axis);
    const auto count = length == 0 ? 1 : (length + split_size - 1) / split_size;
    Shape sizes(static_cast<std::size_t>(count), split_size);
    sizes.back() = length - (count - 1) * split_size;
    return split(std::move(sizes), axis);
}

std::vector<Tensor> Tensor::split(Shape sizes, Index axis) const {
    std::vector<Tensor> result;
    result.reserve(sizes.size());
    Index start = 0;
    for (auto length : sizes) {
        result.push_back(narrow(axis, start, length));
        start += length;
    }
    return result;
}

std::vector<Tensor> Tensor::chunk(Index chunks, Index axis) const {
    const auto length = size(axis);
    return length == 0 ? split(Shape(static_cast<std::size_t>(chunks), 0), axis)
                       : split((length + chunks - 1) / chunks, axis);
}

std::vector<Tensor> Tensor::unbind(Index axis) const {
    std::vector<Tensor> result;
    result.reserve(static_cast<std::size_t>(size(axis)));
    for (Index index = 0; index < size(axis); ++index) {
        result.push_back(select(axis, index));
    }
    return result;
}

Tensor Tensor::repeat(Shape repeats) const {
    const auto leading = repeats.size() - shape().size();
    for (std::size_t axis = leading; axis < repeats.size(); ++axis) {
        repeats[axis] *= shape()[axis - leading];
    }
    return detail::record(detail::numeric::repeat(detail::data_of(*this), std::move(repeats)),
        {detail::gradient_edge(*this)}, [shape = shape()](const auto& grad, auto& sink, const auto& edges) {
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::repeat_backward_into(target.value, grad);
        });
}

} // namespace dl
