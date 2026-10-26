#include "dl/ops.h"
#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl {

Tensor cumsum(const Tensor& input, Index axis) {
    axis = detail::axis_index(axis, input.ndim());
    return detail::record(detail::numeric::cumsum(detail::data_of(input), axis), {detail::gradient_edge(input)},
        [axis](const auto& grad, auto& sink, const auto& edges) {
            sink.add_owned(edges[0], detail::numeric::cumsum(grad, axis, true));
        });
}

Tensor argsort(const Tensor& input, Index axis, bool descending) {
    return detail::TensorAccess::wrap(detail::numeric::argsort(detail::data_of(input),
        detail::axis_index(axis, input.ndim()), descending));
}

SortResult sort(const Tensor& input, Index axis, bool descending) {
    auto indices = argsort(input, axis, descending);
    return {gather(input, axis, indices), std::move(indices)};
}

SortResult topk(const Tensor& input, Index k, Index axis, bool largest) {
    auto indices = argsort(input, axis, largest).narrow(axis, 0, k);
    return {gather(input, axis, indices), std::move(indices)};
}

Tensor pad(const Tensor& input, Shape padding, double value) {
    auto shape = input.shape();
    for (std::size_t i = 0; i < padding.size() / 2; ++i)
        shape[shape.size() - 1 - i] += padding[2*i] + padding[2*i+1];
    auto result = detail::numeric::full(shape, value, input.dtype(), input.device());
    auto interior = [&](auto data) {
        for (std::size_t i = 0; i < padding.size() / 2; ++i) {
            const auto axis = input.ndim() - 1 - static_cast<Index>(i);
            data = detail::numeric::slice(data, axis, padding[2*i], padding[2*i] + input.size(axis), 1);
        }
        return data;
    };
    auto target = interior(result);
    detail::numeric::copy_into(target, detail::data_of(input));
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [padding = std::move(padding), original = input.shape()](const auto& grad, auto& sink, const auto& edges) {
            auto center = grad;
            for (std::size_t i = 0; i < padding.size() / 2; ++i) {
                const auto axis = static_cast<Index>(original.size() - 1 - i);
                center = detail::numeric::slice(center, axis, padding[2*i], padding[2*i] + original[axis], 1);
            }
            sink.add(edges[0], center);
        });
}

Tensor one_hot(const Tensor& indices, Index classes, DType dtype) {
    auto categories = arange(static_cast<double>(classes), {.dtype = DType::i64, .device = indices.device()});
    return eq(indices.unsqueeze(-1), categories).astype(dtype);
}

} // namespace dl
