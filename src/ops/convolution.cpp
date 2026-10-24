#include "dl/ops.h"
#include "dl/amp.h"

#include "autograd/record.h"
#include "backends/backend.h"
#include "core/spatial.h"
#include "ops/numeric.h"

namespace dl {
namespace {

namespace numeric = detail::numeric;

/// 两个后端交付独立的 NCHW 结果，通道偏置和对应归约在同一个节点中处理。
template <class Backward>
Tensor record_convolution(detail::TensorData result, std::vector<detail::Edge> edges,
                          const std::optional<Tensor>& bias, Backward backward) {
    const auto channels = result.layout.sizes[1];
    if (bias) {
        auto offset = numeric::reshape(numeric::cast(detail::data_of(*bias), result.dtype), {1, channels, 1, 1});
        numeric::add_into(result, numeric::expand(offset, result.layout.sizes));
    }
    return detail::record(std::move(result), std::move(edges),
        [backward = std::move(backward), channels](const auto& grad, auto& sink, const auto& inputs) {
            if (sink.needs(inputs[0]) || sink.needs(inputs[1])) backward(grad, sink, inputs);
            if (sink.needs(inputs[2])) {
                auto target = sink.target(inputs[2], {channels}, grad.dtype, grad.device());
                auto channel_view = numeric::reshape(target.value, {1, channels, 1, 1});
                // N/H/W 直接归约到一维偏置的共享目标视图，已有贡献继续相加。
                numeric::sum_into(channel_view, grad, {0, 2, 3}, target.accumulate);
            }
        });
}

Tensor convolution_gemm(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, Shape output_shape,
                        Conv2dOptions options, DType compute_dtype) {
    const Size2 kernel{weight.size(2), weight.size(3)};
    const Shape filters{options.groups, weight.size(0) / options.groups,
                         weight.size(1) * kernel.height * kernel.width};
    const Shape grouped_output{input.size(0), options.groups, weight.size(0) / options.groups,
                                output_shape[2] * output_shape[3]};
    const bool pointwise = kernel.height == 1 && kernel.width == 1
        && options.stride.height == 1 && options.stride.width == 1
        && options.padding.height == 0 && options.padding.width == 0;
    // 单点窗口按通道和空间轴重排，能表示为视图时直接复用输入存储。
    auto columns = [&] {
        auto prepared_input = numeric::cast(detail::data_of(input), compute_dtype);
        return pointwise
            ? numeric::reshape(prepared_input, {input.size(0), options.groups, filters[2], grouped_output[3]})
            : numeric::im2col(prepared_input, kernel, options);
    }();
    auto matrices = numeric::reshape(numeric::cast(detail::data_of(weight), compute_dtype), filters);
    auto result = numeric::reshape(numeric::matmul(matrices, columns), std::move(output_shape));
    std::vector edges{detail::gradient_edge(input), detail::gradient_edge(weight),
        bias ? detail::gradient_edge(*bias) : detail::Edge{}};
    std::optional<detail::SavedTensor> saved_columns = edges[1].node ? std::optional{columns} : std::nullopt;
    std::optional<detail::SavedTensor> saved_filters = edges[0].node ? std::optional{matrices} : std::nullopt;
    return record_convolution(std::move(result), std::move(edges), bias,
        [saved_columns = std::move(saved_columns), saved_filters = std::move(saved_filters),
         input_shape = input.shape(), weight_shape = weight.shape(), grouped_output, kernel, options, compute_dtype, pointwise]
        (const auto& grad, auto& sink, const auto& inputs) {
            auto grouped = numeric::reshape(numeric::cast(grad, compute_dtype), grouped_output);
            // 单点卷积直接写入梯度的矩阵视图，其余几何经 col2im 汇回同一目标。
            if (sink.needs(inputs[0])) {
                auto target = sink.target(inputs[0], input_shape, compute_dtype, grad.device());
                auto transposed_filters = numeric::transpose(*saved_filters, 1, 2);
                if (pointwise) {
                    auto matrix_target = numeric::reshape(target.value,
                        {input_shape[0], options.groups, weight_shape[1], grouped_output[3]});
                    numeric::matmul_into(matrix_target, transposed_filters, grouped, target.accumulate);
                } else {
                    auto columns_grad = numeric::matmul(transposed_filters, grouped);
                    numeric::col2im_into(target.value, columns_grad, kernel, options, target.accumulate);
                }
            }
            if (sink.needs(inputs[1])) {
                auto batches = numeric::matmul(grouped, numeric::transpose(*saved_columns, 2, 3));
                auto target = sink.target(inputs[1], weight_shape, batches.dtype, batches.device());
                numeric::sum_into(target.value, batches, {0}, target.accumulate);
            }
        });
}

Tensor convolution_library(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias, Shape output_shape,
                            Conv2dOptions options, DType compute_dtype, detail::ConvolutionBackend backend) {
    using detail::ConvolutionPass;
    auto prepared_input = backend.prepare(detail::data_of(input), compute_dtype);
    auto prepared_weight = backend.prepare(detail::data_of(weight), compute_dtype);
    std::vector edges{detail::gradient_edge(input), detail::gradient_edge(weight),
        bias ? detail::gradient_edge(*bias) : detail::Edge{}};
    std::optional<detail::SavedTensor> image = edges[1].node ? std::optional{prepared_input} : std::nullopt;
    std::optional<detail::SavedTensor> filters = edges[0].node ? std::optional{prepared_weight} : std::nullopt;
    auto result = backend.execute(ConvolutionPass::forward, prepared_input, prepared_weight,
                          std::move(output_shape), options);
    return record_convolution(std::move(result), std::move(edges), bias,
        [image = std::move(image), filters = std::move(filters), input_shape = input.shape(),
         weight_shape = weight.shape(), options, compute_dtype, backend](const auto& grad, auto& sink, const auto& inputs) {
            const auto seed = backend.prepare(grad, compute_dtype);
            // 两种 VJP 各自只读取另一侧的前向数据，目标形状由原始输入给出。
            if (sink.needs(inputs[0])) {
                sink.add_owned(inputs[0], backend.execute(ConvolutionPass::input_gradient, seed, *filters, input_shape, options));
            }
            if (sink.needs(inputs[1])) {
                sink.add_owned(inputs[1], backend.execute(ConvolutionPass::weight_gradient, seed, *image, weight_shape, options));
            }
        });
}

} // namespace

Tensor conv2d(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias, Conv2dOptions options) {
    const auto compute_dtype = input.dtype() == DType::f32 && is_autocast_enabled(input.device().type)
        ? autocast_dtype(input.device().type) : input.dtype();
    const auto geometry = detail::spatial_geometry(input.shape(), {weight.size(2), weight.size(3)},
                                                   options.stride, options.padding, options.dilation);
    Shape output_shape{input.size(0), weight.size(0), geometry.output_height, geometry.output_width};
    const auto backend = detail::backend(input.device()).convolution;
    return backend.execute ? convolution_library(input, weight, bias, std::move(output_shape), options, compute_dtype, backend)
                           : convolution_gemm(input, weight, bias, std::move(output_shape), options, compute_dtype);
}

} // namespace dl
