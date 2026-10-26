#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

#include <cmath>
#include <limits>

namespace dl {
namespace {

namespace numeric = detail::numeric;
using detail::data_of;

Index group_size(const Tensor& input, const Shape& axes) {
    Index count = 1;
    for (auto axis : axes) count *= input.size(axis);
    return count;
}

detail::TensorData constant(const detail::TensorData& input, double value) {
    return numeric::full({}, value, input.dtype, input.device());
}

Tensor zero_gradient(detail::TensorData result, const Tensor& input) {
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [shape = input.shape()](const auto& grad, auto& sink, const auto& edges) {
            sink.add_owned(edges[0], numeric::full(shape, 0.0, grad.dtype, grad.device()));
        });
}

Tensor variance(const Tensor& input, Shape axes, double correction, KeepDim keep, bool deviation) {
    axes = detail::normalize_axes(std::move(axes), input.ndim());
    const auto count = group_size(input, axes);
    const double denominator = static_cast<double>(count) - correction;
    auto [result, centered] = numeric::variance(data_of(input), axes, correction, keep);
    if (deviation) result = numeric::unary(result, numeric::Unary::sqrt);
    auto kept_shape = numeric::reduced_shape(input.shape(), axes, KeepDim::yes);
    auto saved = result;
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [centered = std::move(centered), saved = std::move(saved), kept_shape = std::move(kept_shape),
         denominator, deviation](const auto& grad, auto& sink, const auto& edges) {
            auto contribution = numeric::mul(centered, numeric::reshape(grad, kept_shape));
            if (deviation) {
                auto sigma = numeric::reshape(saved, kept_shape);
                auto zero = constant(sigma, 0.0);
                contribution = numeric::where(numeric::compare(sigma, zero, numeric::Compare::eq), zero,
                    numeric::div(contribution, numeric::scale(sigma, denominator)));
            } else {
                contribution = numeric::scale(contribution, 2.0 / denominator);
            }
            sink.add_owned(edges[0], std::move(contribution));
        });
}

} // namespace

Tensor var(const Tensor& input, double correction) {
    return var(input, detail::all_axes(input.ndim()), correction);
}
Tensor var(const Tensor& input, Shape axes, double correction, KeepDim keep) {
    return variance(input, std::move(axes), correction, keep, false);
}
Tensor stddev(const Tensor& input, double correction) {
    return stddev(input, detail::all_axes(input.ndim()), correction);
}
Tensor stddev(const Tensor& input, Shape axes, double correction, KeepDim keep) {
    return variance(input, std::move(axes), correction, keep, true);
}

Tensor logsumexp(const Tensor& input) { return logsumexp(input, detail::all_axes(input.ndim())); }
Tensor logsumexp(const Tensor& input, Shape axes, KeepDim keep) {
    axes = detail::normalize_axes(std::move(axes), input.ndim());
    if (axes.empty()) return input.clone();
    auto maximum = numeric::amax(data_of(input), axes, KeepDim::yes);
    auto infinite = numeric::compare(numeric::unary(maximum, numeric::Unary::abs),
        constant(maximum, std::numeric_limits<double>::infinity()), numeric::Compare::eq);
    // 全为负无穷的组使用零平移，指数和自然为零，对数结果仍为负无穷。
    auto shift = numeric::where(infinite, constant(maximum, 0.0), maximum);
    auto shifted = numeric::sub(data_of(input), shift);
    auto normalizer = numeric::unary(numeric::sum(numeric::unary(shifted, numeric::Unary::exp),
        axes, KeepDim::yes), numeric::Unary::log);
    auto result = numeric::reshape(numeric::add(normalizer, shift), numeric::reduced_shape(input.shape(), axes, keep));
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [shifted = std::move(shifted), normalizer = std::move(normalizer)]
        (const auto& grad, auto& sink, const auto& edges) {
            auto probabilities = numeric::unary(numeric::sub(shifted, normalizer), numeric::Unary::exp);
            sink.add_owned(edges[0], numeric::mul(probabilities, numeric::reshape(grad, normalizer.layout.sizes)));
        });
}

Tensor vector_norm(const Tensor& input, double order) {
    return vector_norm(input, detail::all_axes(input.ndim()), order);
}
Tensor vector_norm(const Tensor& input, Shape axes, double order, KeepDim keep) {
    axes = detail::normalize_axes(std::move(axes), input.ndim());
    if (order == 1.0) return sum(abs(input), std::move(axes), keep);
    if (std::isinf(order)) {
        return order > 0.0 ? amax(abs(input), std::move(axes), keep)
                           : amin(abs(input), std::move(axes), keep);
    }
    const auto& source = data_of(input);
    auto zero = constant(source, 0.0);
    if (order == 0.0) {
        auto nonzero = numeric::cast(numeric::compare(source, zero, numeric::Compare::ne), source.dtype);
        return zero_gradient(numeric::sum(nonzero, axes, keep), input);
    }
    if (group_size(input, axes) == 0) {
        return zero_gradient(numeric::full(numeric::reduced_shape(input.shape(), axes, keep),
            0.0, input.dtype(), input.device()), input);
    }
    auto magnitude = numeric::unary(source, numeric::Unary::abs);
    auto maximum = numeric::amax(magnitude, axes, KeepDim::yes);
    auto scale = numeric::where(numeric::compare(maximum, zero, numeric::Compare::eq), constant(source, 1.0), maximum);
    // 先按组缩放绝对值，幂和在归一化量级上计算，最后恢复原来的尺度。
    auto powered = numeric::pow(numeric::div(magnitude, scale), constant(source, order));
    auto norm = numeric::mul(scale, numeric::pow(numeric::sum(powered, axes, KeepDim::yes), constant(source, 1.0 / order)));
    auto infinity = constant(source, std::numeric_limits<double>::infinity());
    norm = numeric::where(numeric::compare(maximum, infinity, numeric::Compare::eq), maximum, norm);
    auto result = numeric::reshape(norm, numeric::reduced_shape(input.shape(), axes, keep));
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [source, magnitude = std::move(magnitude), norm = std::move(norm), order]
        (const auto& grad, auto& sink, const auto& edges) {
            auto zero = constant(source, 0.0);
            auto slope = order == 2.0 ? numeric::div(source, norm)
                : numeric::mul(numeric::div(source, magnitude),
                    numeric::pow(numeric::div(magnitude, norm), constant(source, order - 1.0)));
            auto contribution = numeric::mul(slope, numeric::reshape(grad, norm.layout.sizes));
            sink.add_owned(edges[0], numeric::where(numeric::compare(magnitude, zero, numeric::Compare::eq), zero, contribution));
        });
}

} // namespace dl
