#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl {

Tensor conv1d(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias, Conv1dOptions options) {
    const bool unbatched = input.ndim() == 2;
    auto batched = unbatched ? input.unsqueeze(0) : input;
    auto result = conv2d(batched.unsqueeze(2), weight.unsqueeze(2), std::move(bias),
        {{1, options.stride}, {0, options.padding}, {1, options.dilation}, options.groups}).squeeze(2);
    return unbatched ? result.squeeze(0) : result;
}

Tensor max_pool1d(const Tensor& input, Pool1dOptions options) {
    const bool unbatched = input.ndim() == 2;
    auto batched = unbatched ? input.unsqueeze(0) : input;
    auto result = max_pool2d(batched.unsqueeze(2),
        {{1, options.kernel}, {1, options.stride}, {0, options.padding}}).squeeze(2);
    return unbatched ? result.squeeze(0) : result;
}

Tensor avg_pool1d(const Tensor& input, Pool1dOptions options, bool count_include_pad) {
    const bool unbatched = input.ndim() == 2;
    auto batched = unbatched ? input.unsqueeze(0) : input;
    auto result = avg_pool2d(batched.unsqueeze(2),
        {{1, options.kernel}, {1, options.stride}, {0, options.padding}}, count_include_pad).squeeze(2);
    return unbatched ? result.squeeze(0) : result;
}

Tensor max_pool2d(const Tensor& input, Pool2dOptions options) {
    auto edge = detail::gradient_edge(input);
    auto result = detail::numeric::max_pool2d(detail::data_of(input), options, bool(edge.node));
    return detail::record(std::move(result.values), {std::move(edge)},
        [selected = std::move(result.indices), shape = input.shape()]
        (const auto& grad, auto& sink, const auto& edges) {
            // 选中位置可能重叠；首次目标置零，其他分支已写入的贡献继续保留。
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::max_pool2d_backward_into(target.value, grad, *selected);
        });
}

Tensor avg_pool2d(const Tensor& input, Pool2dOptions options, bool count_include_pad) {
    return detail::record(detail::numeric::avg_pool2d(detail::data_of(input), options, count_include_pad),
        {detail::gradient_edge(input)},
        [shape = input.shape(), options, count_include_pad](const auto& grad, auto& sink, const auto& edges) {
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            detail::numeric::avg_pool2d_backward_into(target.value, grad, options, count_include_pad);
        });
}

} // namespace dl
