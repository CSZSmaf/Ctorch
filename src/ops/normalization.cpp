#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

#include <limits>

namespace dl {
namespace {

namespace numeric = detail::numeric;

numeric::NormalizationGradients normalization_targets(detail::GradSink& sink,
    const std::vector<detail::Edge>& edges, const detail::TensorData& grad, const Shape& parameter_shape) {
    numeric::NormalizationGradients result;
    std::array outputs{&result.input, &result.weight, &result.bias};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (!sink.needs(edges[i])) continue;
        auto target = sink.target(edges[i], i == 0 ? grad.layout.sizes : parameter_shape, grad.dtype, grad.device());
        *outputs[i] = target.value;
        result.accumulate[i] = target.accumulate;
    }
    return result;
}

Tensor normalize_batch(const Tensor& input, const std::optional<BatchNormState>& state,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias,
    bool training, double momentum, double epsilon) {
    using detail::data_of;
    using detail::gradient_edge;
    const auto channels = input.size(1);
    const auto dtype = accumulation_dtype(input.dtype());
    const bool batch_statistics = training || !state;
    std::vector edges{gradient_edge(input), weight ? gradient_edge(*weight) : detail::Edge{},
                      bias ? gradient_edge(*bias) : detail::Edge{}};
    auto weight_data = weight ? std::optional{numeric::cast(data_of(*weight), dtype)} : std::nullopt;
    auto bias_data = bias ? std::optional{numeric::cast(data_of(*bias), dtype)} : std::nullopt;
    auto running = state ? std::optional{numeric::BatchNormalizationState{data_of(state->mean), data_of(state->variance)}}
                         : std::nullopt;
    auto result = numeric::batch_normalization(data_of(input), weight_data, bias_data,
        std::move(running), training, momentum, epsilon);

    if (!edges[0].node && !edges[1].node) {
        Shape axes{0};
        for (Index axis = 2; axis < input.ndim(); ++axis) axes.push_back(axis);
        return detail::record(std::move(result.output), std::move(edges),
            [axes = std::move(axes), channels](const auto& grad, auto& sink, const auto& inputs) {
                auto target = sink.target(inputs[2], {channels}, grad.dtype, grad.device());
                numeric::sum_into(target.value, grad, axes, target.accumulate);
            });
    }

    // 统计量保存为通道向量；eval 的输入梯度只需要逆标准差与权重。
    const bool save_source = bool(edges[1].node) || (batch_statistics && bool(edges[0].node));
    auto source = save_source ? std::optional{data_of(input)} : std::nullopt;
    if (!edges[0].node) weight_data.reset();
    return detail::record(std::move(result.output), std::move(edges),
        [source = std::move(source), statistics = std::move(*result.statistics),
         weight_data = std::move(weight_data), channels, batch_statistics]
        (const auto& grad, auto& sink, const auto& inputs) {
            auto targets = normalization_targets(sink, inputs, grad, {channels});
            numeric::batch_normalization_backward(grad, source ? *source : grad,
                statistics, weight_data, batch_statistics, targets);
        });
}

Tensor normalize(const Tensor& input, const Shape& normalized_shape, std::optional<Tensor> weight,
                 std::optional<Tensor> bias, double epsilon, numeric::NormalizationKind kind) {
    using detail::data_of;
    using detail::gradient_edge;
    std::vector edges{gradient_edge(input), weight ? gradient_edge(*weight) : detail::Edge{},
                      bias ? gradient_edge(*bias) : detail::Edge{}};
    const auto dtype = accumulation_dtype(input.dtype());
    auto weight_data = weight ? std::optional{numeric::cast(data_of(*weight), dtype)} : std::nullopt;
    auto bias_data = bias ? std::optional{numeric::cast(data_of(*bias), dtype)} : std::nullopt;
    const bool save_statistics = bool(edges[0].node) || bool(edges[1].node);
    auto result = numeric::normalization(data_of(input), weight_data, bias_data,
        detail::numel(normalized_shape), epsilon, kind, save_statistics);

    // 仅偏置求导时只需对上游梯度求和，前向输入和统计量可在调用结束后释放。
    if (!save_statistics) {
        return detail::record(std::move(result.output), std::move(edges),
            [normalized_shape](const auto& grad, auto& sink, const auto& inputs) {
                sink.add_reduced(inputs[2], grad, normalized_shape);
            });
    }

    // 输入梯度需要仿射权重；权重梯度只依赖原输入及每行统计量。
    if (!edges[0].node) weight_data.reset();
    return detail::record(std::move(result.output), std::move(edges),
        [source = data_of(input), statistics = std::move(*result.statistics),
         weight_data = std::move(weight_data), normalized_shape, kind]
        (const auto& grad, auto& sink, const auto& inputs) {
            auto targets = normalization_targets(sink, inputs, grad, normalized_shape);
            numeric::normalization_backward(grad, source, statistics, weight_data,
                detail::numel(normalized_shape), kind, targets);
        });
}

} // namespace

BatchNormResult batch_norm(const Tensor& input, std::optional<BatchNormState> state,
    std::optional<Tensor> weight, std::optional<Tensor> bias, bool training, double momentum, double epsilon) {
    if (state) {
        // 函数式调用更新独立状态，原地调用由下面的入口直接使用模块缓冲。
        state = BatchNormState{state->mean.detach(), state->variance.detach()};
        if (training) state = BatchNormState{state->mean.clone(), state->variance.clone()};
    }
    auto output = normalize_batch(input, state, weight, bias, training, momentum, epsilon);
    return {std::move(output), std::move(state)};
}

Tensor batch_norm_(const Tensor& input, BatchNormState state, std::optional<Tensor> weight,
    std::optional<Tensor> bias, bool training, double momentum, double epsilon) {
    return normalize_batch(input, state, weight, bias, training, momentum, epsilon);
}

Tensor layer_norm(const Tensor& input, const Shape& normalized_shape,
                  std::optional<Tensor> weight, std::optional<Tensor> bias, double epsilon) {
    return normalize(input, normalized_shape, std::move(weight), std::move(bias), epsilon,
                     numeric::NormalizationKind::layer);
}

Tensor group_norm(const Tensor& input, Index groups, std::optional<Tensor> weight,
                  std::optional<Tensor> bias, double epsilon) {
    Index group_elements = input.size(1) / groups;
    for (Index axis = 2; axis < input.ndim(); ++axis) group_elements *= input.size(axis);
    auto grouped = input.reshape({input.size(0), groups, group_elements});
    auto result = layer_norm(grouped, {group_elements}, std::nullopt, std::nullopt, epsilon).reshape(input.shape());
    Shape affine_shape(static_cast<std::size_t>(input.ndim()), 1);
    affine_shape[1] = input.size(1);
    if (weight) result = result * weight->reshape(affine_shape);
    return bias ? result + bias->reshape(affine_shape) : result;
}

Tensor rms_norm(const Tensor& input, const Shape& normalized_shape,
                std::optional<Tensor> weight, std::optional<double> epsilon) {
    const double eps = epsilon.value_or(input.dtype() == DType::f64
        ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon());
    return normalize(input, normalized_shape, std::move(weight), std::nullopt, eps,
                     numeric::NormalizationKind::rms);
}

} // namespace dl
