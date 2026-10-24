#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl {
namespace {

template <class Forward, class Backward>
Tensor image_operation(const Tensor& input, Forward forward, Backward backward) {
    return detail::record(forward(detail::data_of(input)), {detail::gradient_edge(input)},
        [shape = input.shape(), backward](const auto& grad, auto& sink, const auto& edges) {
            // 重叠区域和插值邻点向同一目标追加贡献，已有分支保持原累计值。
            auto target = sink.target(edges[0], shape, grad.dtype, grad.device(), detail::GradInit::zero);
            backward(target.value, grad);
        });
}

} // namespace

Tensor adaptive_avg_pool2d(const Tensor& input, Size2 output_size) {
    return image_operation(input,
        [output_size](const auto& source) { return detail::numeric::adaptive_avg_pool2d(source, output_size); },
        [](auto& target, const auto& grad) { detail::numeric::adaptive_avg_pool2d_backward_into(target, grad); });
}

Tensor adaptive_avg_pool1d(const Tensor& input, Index output_size) {
    return adaptive_avg_pool2d(input.unsqueeze(-2), {1, output_size}).squeeze(-2);
}

Tensor interpolate(const Tensor& input, Size2 output_size, InterpolationMode mode, bool align_corners) {
    return image_operation(input,
        [=](const auto& source) { return detail::numeric::interpolate(source, output_size, mode, align_corners); },
        [=](auto& target, const auto& grad) {
            detail::numeric::interpolate_backward_into(target, grad, mode, align_corners);
        });
}

} // namespace dl
