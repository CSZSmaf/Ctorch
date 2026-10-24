#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl::detail::numeric {

TensorData concat(std::span<const TensorData> inputs, Index axis) {
    auto shape = inputs.front().layout.sizes;
    shape[axis] = 0;
    for (const auto& input : inputs) {
        shape[axis] += input.layout.sizes[axis];
    }
    auto result = empty_data(std::move(shape), inputs.front().dtype, inputs.front().device());
    Index offset = 0;
    for (const auto& input : inputs) {
        const auto end = offset + input.layout.sizes[axis];
        auto destination = slice(result, axis, offset, end, 1);
        copy_into(destination, input);
        offset = end;
    }
    return result;
}

} // namespace dl::detail::numeric

namespace dl {

Tensor concat(std::span<const Tensor> inputs, Index axis) {
    axis = detail::axis_index(axis, inputs.front().ndim());
    std::vector<detail::TensorData> data;
    std::vector<detail::Edge> edges;
    Shape lengths;
    data.reserve(inputs.size());
    edges.reserve(inputs.size());
    lengths.reserve(inputs.size());
    for (const auto& input : inputs) {
        data.push_back(detail::data_of(input));
        edges.push_back(detail::gradient_edge(input));
        lengths.push_back(input.size(axis));
    }
    return detail::record(detail::numeric::concat(data, axis), std::move(edges),
        [lengths = std::move(lengths), axis](const auto& grad, auto& sink, const auto& inputs) {
            // 每个输入对应结果中的一个连续轴区间，反向沿同一区间分配贡献。
            Index offset = 0;
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                const auto end = offset + lengths[index];
                if (sink.needs(inputs[index])) {
                    sink.add(inputs[index], detail::numeric::slice(grad, axis, offset, end, 1));
                }
                offset = end;
            }
        });
}

Tensor stack(std::span<const Tensor> inputs, Index axis) {
    axis = detail::axis_index(axis, inputs.front().ndim() + 1);
    std::vector<Tensor> expanded;
    expanded.reserve(inputs.size());
    for (const auto& input : inputs) {
        expanded.push_back(input.unsqueeze(axis));
    }
    return concat(expanded, axis);
}

} // namespace dl
