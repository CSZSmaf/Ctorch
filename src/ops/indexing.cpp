#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl::detail::numeric {
namespace {

TensorData expanded_indices(const TensorData& indices, const Shape& shape, Index axis) {
    Shape selected_shape(shape.size(), 1);
    selected_shape[axis] = indices.layout.numel();
    return expand(reshape(indices, std::move(selected_shape)), shape);
}

} // namespace

TensorData index_select(const TensorData& input, Index axis, const TensorData& indices) {
    auto shape = input.layout.sizes;
    shape[axis] = indices.layout.numel();
    return gather(input, axis, expanded_indices(indices, shape, axis));
}

void index_backward_into(TensorData& target, const TensorData& grad, Index axis, const TensorData& indices) {
    scatter_add_into(target, axis, expanded_indices(indices, grad.layout.sizes, axis), grad);
}

TensorData scatter_add(const TensorData& input, Index axis, const TensorData& indices, const TensorData& source) {
    const auto dtype = accumulation_dtype(input.dtype);
    auto result = input.dtype == dtype ? clone(input) : cast(input, dtype);
    scatter_add_into(result, axis, indices, cast(source, dtype));
    return result;
}

} // namespace dl::detail::numeric

namespace dl {

Tensor embedding(const Tensor& indices, const Tensor& weight) {
    auto shape = indices.shape();
    shape.push_back(weight.size(1));
    return index_select(weight, 0, indices.reshape({indices.numel()})).reshape(std::move(shape));
}

Tensor index_select(const Tensor& input, Index axis, const Tensor& indices) {
    axis = detail::axis_index(axis, input.ndim());
    return detail::record(detail::numeric::index_select(detail::data_of(input), axis, detail::data_of(indices)),
        {detail::gradient_edge(input)},
        [shape = input.shape(), axis, selected = detail::data_of(indices)]
        (const auto& grad, auto& sink, const auto& edges) {
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::index_backward_into(target.value, grad, axis, selected);
        });
}

Tensor gather(const Tensor& input, Index axis, const Tensor& indices) {
    axis = detail::axis_index(axis, input.ndim());
    return detail::record(detail::numeric::gather(detail::data_of(input), axis, detail::data_of(indices)),
        {detail::gradient_edge(input)},
        [shape = input.shape(), axis, selected = detail::data_of(indices)]
        (const auto& grad, auto& sink, const auto& edges) {
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::scatter_add_into(target.value, axis, selected, grad);
        });
}

Tensor scatter_add(const Tensor& input, Index axis, const Tensor& indices, const Tensor& source) {
    axis = detail::axis_index(axis, input.ndim());
    return detail::record(detail::numeric::scatter_add(detail::data_of(input), axis,
            detail::data_of(indices), detail::data_of(source)),
        {detail::gradient_edge(input), detail::gradient_edge(source)},
        [axis, selected = detail::data_of(indices), source_shape = source.shape()]
        (const auto& grad, auto& sink, const auto& edges) {
            if (sink.needs(edges[0])) {
                sink.add(edges[0], grad);
            }
            if (sink.needs(edges[1])) {
                auto contribution = detail::numeric::gather(grad, axis, selected);
                if (contribution.layout.sizes == source_shape) {
                    sink.add_owned(edges[1], std::move(contribution));
                } else {
                    // source 的有效前缀写入目标，其余位置保留零值或其他分支的贡献。
                    auto target = sink.target(edges[1], source_shape, grad.dtype, grad.device(), detail::GradInit::zero);
                    auto prefix = target.value;
                    prefix.layout.sizes = contribution.layout.sizes;
                    if (target.accumulate) detail::numeric::add_into(prefix, contribution);
                    else detail::numeric::copy_into(prefix, contribution);
                }
            }
        });
}

} // namespace dl
