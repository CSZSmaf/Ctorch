#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

#include <limits>

namespace dl {
namespace {

namespace numeric = detail::numeric;

struct BatchComputation {
    Tensor output;
    std::optional<BatchNormState> batch;
};

BatchComputation normalize_batch(const Tensor& input, const std::optional<BatchNormState>& state,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, bool training, double epsilon) {
    using detail::data_of;
    using detail::gradient_edge;
    const auto channels = input.size(1);
    const auto count = input.numel() / channels;
    const auto dtype = accumulation_dtype(input.dtype());
    Shape broadcast(static_cast<std::size_t>(input.ndim()), 1);
    broadcast[1] = channels;
    Shape axes{0};
    for (Index axis = 2; axis < input.ndim(); ++axis) axes.push_back(axis);
    const bool batch_statistics = training || !state;
    auto average = [&](const auto& values) {
        return numeric::scale(numeric::sum(values, axes, KeepDim::yes), 1.0 / static_cast<double>(count));
    };
    std::vector edges{gradient_edge(input), weight ? gradient_edge(*weight) : detail::Edge{},
                      bias ? gradient_edge(*bias) : detail::Edge{}};
    // 工作值沿前向顺序替换，类型展开和中心化缓冲在最后使用后释放。
    auto output = numeric::cast(data_of(input), dtype);
    auto center = batch_statistics ? average(output) : numeric::reshape(data_of(state->mean), broadcast);
    output = numeric::sub(output, center);
    auto variance = batch_statistics ? average(numeric::mul(output, output))
                                    : numeric::reshape(data_of(state->variance), broadcast);
    auto inverse = numeric::div(numeric::full({}, 1.0, dtype, input.device()),
        numeric::unary(numeric::add(variance, numeric::full({}, epsilon, dtype, input.device())), numeric::Unary::sqrt));
    auto weight_data = weight ? std::optional{numeric::reshape(numeric::cast(data_of(*weight), dtype), broadcast)}
                              : std::nullopt;
    output = numeric::mul(output, inverse);
    if (weight_data) output = numeric::mul(output, *weight_data);
    if (bias) {
        auto offset = numeric::reshape(numeric::cast(data_of(*bias), dtype), broadcast);
        numeric::add_into(output, numeric::expand(offset, input.shape()));
    }

    // 运行状态读取同一次前向的总体统计，数据句柄独立于后面的求导节点。
    std::optional<BatchNormState> batch;
    if (training && state) {
        batch = BatchNormState{detail::make_tensor(numeric::reshape(center, {channels})),
            detail::make_tensor(numeric::reshape(variance, {channels}))};
    }

    if (!edges[0].node && !edges[1].node) {
        auto result = detail::record(std::move(output), std::move(edges),
            [axes = std::move(axes), channels](const auto& grad, auto& sink, const auto& inputs) {
                auto target = sink.target(inputs[2], {channels}, grad.dtype, grad.device());
                numeric::sum_into(target.value, grad, axes, target.accumulate);
            });
        return {std::move(result), std::move(batch)};
    }

    // eval 的输入梯度只依赖逆标准差和仿射权重；其他方向重算标准化输入。
    const bool save_source = bool(edges[1].node) || (batch_statistics && bool(edges[0].node));
    auto saved_source = save_source ? std::optional{data_of(input)} : std::nullopt;
    auto saved_center = save_source ? std::optional{batch_statistics ? center : numeric::clone(center)} : std::nullopt;
    if (!edges[0].node) weight_data.reset();
    auto result = detail::record(std::move(output), std::move(edges),
        [source = std::move(saved_source), center = std::move(saved_center), inverse = std::move(inverse),
         weight_data = std::move(weight_data), axes = std::move(axes), channels, count, batch_statistics]
        (const auto& grad, auto& sink, const auto& inputs) {
            const bool input_gradient = sink.needs(inputs[0]);
            const bool weight_gradient = sink.needs(inputs[1]);
            const bool bias_gradient = sink.needs(inputs[2]);
            const bool coupled_input = batch_statistics && input_gradient;
            std::optional<detail::TensorData> normalized, total, projection;
            if (coupled_input || weight_gradient) {
                normalized = numeric::mul(numeric::sub(*source, *center), inverse);
                projection = numeric::sum(numeric::mul(grad, *normalized), axes, KeepDim::yes);
            }
            if (coupled_input || bias_gradient) total = numeric::sum(grad, axes, KeepDim::yes);
            if (input_gradient) {
                auto factor = weight_data ? numeric::mul(inverse, *weight_data) : inverse;
                if (coupled_input) {
                    // 两个通道归约同时服务输入与仿射梯度，修正量直接追加到独立贡献。
                    auto contribution = numeric::clone(grad);
                    const double inverse_count = 1.0 / static_cast<double>(count);
                    numeric::axpy(contribution, numeric::expand(*total, grad.layout.sizes), -inverse_count);
                    numeric::axpy(contribution, numeric::mul(*normalized, *projection), -inverse_count);
                    sink.add_owned(inputs[0], numeric::mul(contribution, factor));
                } else {
                    sink.add_owned(inputs[0], numeric::mul(grad, factor));
                }
            }
            if (weight_gradient) sink.add_owned(inputs[1], numeric::reshape(*projection, {channels}));
            if (bias_gradient) sink.add_owned(inputs[2], numeric::reshape(*total, {channels}));
        });
    return {std::move(result), std::move(batch)};
}

void update_batch_state(BatchNormState state, const BatchNormState& batch, Index count, double momentum) {
    const auto samples = static_cast<double>(count);
    numeric::momentum(detail::TensorAccess::mutable_data(state.mean), detail::data_of(batch.mean),
        1.0 - momentum, momentum);
    numeric::momentum(detail::TensorAccess::mutable_data(state.variance), detail::data_of(batch.variance),
        1.0 - momentum, momentum * samples / (samples - 1.0));
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
            auto gradients = numeric::normalization_backward(grad, source, statistics, weight_data,
                normalized_shape, kind, {sink.needs(inputs[0]), sink.needs(inputs[1]), sink.needs(inputs[2])});
            if (gradients.input) sink.add_owned(inputs[0], std::move(*gradients.input));
            if (gradients.weight) sink.add_owned(inputs[1], std::move(*gradients.weight));
            if (gradients.bias) sink.add_owned(inputs[2], std::move(*gradients.bias));
        });
}

} // namespace

BatchNormResult batch_norm(const Tensor& input, std::optional<BatchNormState> state,
    std::optional<Tensor> weight, std::optional<Tensor> bias, bool training, double momentum, double epsilon) {
    auto result = normalize_batch(input, state, weight, bias, training, epsilon);
    if (state) {
        // 状态返回值不带梯度历史；训练在独立存储上更新，eval 直接共享只读用途的旧值。
        state = BatchNormState{state->mean.detach(), state->variance.detach()};
        if (training) {
            state = BatchNormState{state->mean.clone(), state->variance.clone()};
            update_batch_state(*state, *result.batch, input.numel() / input.size(1), momentum);
        }
    }
    return {std::move(result.output), std::move(state)};
}

Tensor batch_norm_(const Tensor& input, BatchNormState state, std::optional<Tensor> weight,
    std::optional<Tensor> bias, bool training, double momentum, double epsilon) {
    auto result = normalize_batch(input, state, weight, bias, training, epsilon);
    if (training) update_batch_state(state, *result.batch, input.numel() / input.size(1), momentum);
    return std::move(result.output);
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
