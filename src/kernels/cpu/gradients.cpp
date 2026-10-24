#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"

#include <cmath>

namespace dl::detail::cpu {
namespace {

template <class T, class Function>
void each_value(TensorData& gradient, Function function) {
    auto* values = gradient.mutable_data<T>();
    if (gradient.layout.is_contiguous()) {
        const auto count = gradient.layout.numel();
        for (Index i = 0; i < count; ++i) function(values[i]);
    } else {
        IterationPlan<1> plan(gradient.layout.sizes, {gradient.layout.strides});
        plan.for_each([&](Index, const auto& offset) { function(values[offset[0]]); });
    }
}

} // namespace

TensorData unscale_gradients(std::span<TensorData> gradients, double inverse_scale) {
    bool finite = true;
    for (auto& gradient : gradients) {
        visit_floating(gradient.dtype, [&]<class T> {
            const auto factor = static_cast<T>(inverse_scale);
            each_value<T>(gradient, [&](T& value) {
                value *= factor;
                finite &= std::isfinite(value);
            });
        });
    }
    return full({}, finite ? 0.0 : 1.0, DType::f32, gradients.front().device());
}

TensorData clip_grad_norm(std::span<TensorData> gradients, double max_norm, double order) {
    bool has_values = false;
    for (const auto& gradient : gradients) has_values |= gradient.layout.numel() != 0;
    if (!has_values) return full({}, 0.0, gradients.front().dtype, gradients.front().device());
    auto result = empty_data({}, gradients.front().dtype, gradients.front().device());
    visit_floating(result.dtype, [&]<class T> {
        const bool maximum = std::isinf(order);
        const auto power = static_cast<T>(order);
        T largest = 0;
        if (order != 1.0) {
            for (auto& gradient : gradients) {
                each_value<T>(gradient, [&](T value) {
                    const T absolute = std::abs(value);
                    if (std::isnan(absolute) || absolute > largest) largest = absolute;
                });
            }
        }
        const T scale = largest == T{0} ? T{1} : largest;
        T total = 0;
        // 幂和沿所有梯度共享的尺度累计，保留原范数算法的数值范围。
        if (!maximum) {
            for (auto& gradient : gradients) {
                each_value<T>(gradient, [&](T value) {
                    const T absolute = std::abs(value);
                    const T normalized = absolute / scale;
                    total += order == 1.0 ? absolute : order == 2.0 ? normalized * normalized
                        : std::pow(normalized, power);
                });
            }
        }
        const T norm = maximum ? largest : order == 1.0 ? total : std::isinf(scale) ? scale
            : scale * (order == 2.0 ? std::sqrt(total) : std::pow(total, static_cast<T>(1.0 / order)));
        *result.mutable_data<T>() = norm;
        const T ratio = static_cast<T>(max_norm) / (norm + static_cast<T>(1e-6));
        const T coefficient = ratio < T{0} ? T{0} : ratio > T{1} ? T{1} : ratio;
        for (auto& gradient : gradients) each_value<T>(gradient, [&](T& value) { value *= coefficient; });
    });
    return result;
}

void clip_grad_value(std::span<TensorData> gradients, double bound) {
    for (auto& gradient : gradients) {
        visit_floating(gradient.dtype, [&]<class T> {
            const T limit = static_cast<T>(bound);
            each_value<T>(gradient, [&](T& value) {
                value = value < -limit ? -limit : value > limit ? limit : value;
            });
        });
    }
}

} // namespace dl::detail::cpu
