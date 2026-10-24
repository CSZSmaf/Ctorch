#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dl::detail::cpu {

using numeric::reduced_shape;
namespace {

template <bool Maximum>
void reduce_into(TensorData& result, const TensorData& input, const Shape& axes, bool accumulate) {
    auto kept_shape = reduced_shape(input.layout.sizes, axes, KeepDim::yes);
    auto output_strides = contiguous_strides(kept_shape);
    for (auto axis : axes) {
        output_strides[axis] = 0;
    }
    IterationPlan<2> plan(input.layout.sizes, {input.layout.strides, std::move(output_strides)});
    visit_floating(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        // 首次覆盖填入归约单位元，已有目标从此前汇合的值继续计算。
        if (!accumulate) std::fill_n(target, result.layout.numel(),
            Maximum ? -std::numeric_limits<T>::infinity() : T{0});
        plan.for_each([&](Index, const auto& offsets) {
            auto& accumulated = target[offsets[1]];
            const T value = source[offsets[0]];
            if constexpr (Maximum) {
                // NaN 是最大值归约的吸收值，后续有限值保持已产生的 NaN。
                accumulated = std::isnan(value) ? value : std::max(accumulated, value);
            } else {
                accumulated += value;
            }
        });
    });
}

template <bool Maximum> TensorData reduce(const TensorData& input, const Shape& axes, KeepDim keep) {
    auto result = empty_data(reduced_shape(input.layout.sizes, axes, keep), input.dtype);
    reduce_into<Maximum>(result, input, axes, false);
    return result;
}

} // namespace

TensorData sum(const TensorData& input, const Shape& axes, KeepDim keep) {
    return reduce<false>(input, axes, keep);
}

void sum_into(TensorData& target, const TensorData& input, const Shape& axes, bool accumulate) {
    reduce_into<false>(target, input, axes, accumulate);
}

TensorData amax(const TensorData& input, const Shape& axes, KeepDim keep) {
    return reduce<true>(input, axes, keep);
}

TensorData arg_extreme(const TensorData& input, Index axis, KeepDim keep, bool maximum) {
    auto result = empty_data(reduced_shape(input.layout.sizes, {axis}, keep), DType::i64);
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    visit_dtype(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* output = result.mutable_data<Index>();
        for (Index row = 0; row < result.layout.numel(); ++row) {
            const auto base = (row / inner) * width * inner + row % inner;
            T best = source[input.layout.index_offset(base)];
            Index selected = 0;
            for (Index item = 1; item < width; ++item) {
                const T value = source[input.layout.index_offset(base + item * inner)];
                // 严格比较保留首个并列位置，第一个 NaN 随后保持为所选值。
                if (best == best && (value != value || (maximum ? value > best : value < best))) {
                    best = value;
                    selected = item;
                }
            }
            output[row] = selected;
        }
    });
    return result;
}

namespace {

template <bool Logarithmic> TensorData softmax_forward(const TensorData& input, Index axis) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype));
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    const auto rows = input.layout.numel() / width;
    visit_storage(input.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<A>();
        // 一行只解析一次起始地址，归约轴内沿固定步长访问连续或带步长的输入。
        const auto step = input.layout.strides[axis];
        for (Index row = 0; row < rows; ++row) {
            const auto base = (row / inner) * width * inner + row % inner;
            const auto source_base = input.layout.index_offset(base);
            A maximum = -std::numeric_limits<A>::infinity();
            for (Index item = 0; item < width; ++item) {
                const A value = static_cast<A>(source[source_base + item * step]);
                maximum = std::isnan(value) ? value : std::max(maximum, value);
            }
            A total = 0;
            for (Index item = 0; item < width; ++item) {
                const auto position = base + item * inner;
                const A shifted = static_cast<A>(source[source_base + item * step]) - maximum;
                const A exponential = std::exp(shifted);
                if constexpr (Logarithmic) target[position] = shifted;
                else target[position] = exponential;
                total += exponential;
            }
            // 对数概率直接由平移值减去 log(sum(exp))，微小概率无需先存为零再取对数。
            A log_total = 0;
            if constexpr (Logarithmic) log_total = std::log(total);
            for (Index item = 0; item < width; ++item) {
                auto& value = target[base + item * inner];
                if constexpr (Logarithmic) value -= log_total;
                else value /= total;
            }
        }
    });
    return result;
}

template <bool Logarithmic>
TensorData softmax_gradient(const TensorData& grad, const TensorData& output, Index axis) {
    auto result = empty_data(output.layout.sizes, output.dtype);
    const auto width = output.layout.sizes[axis];
    const auto inner = numel(std::span(output.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    const auto rows = output.layout.numel() / width;
    visit_floating(output.dtype, [&]<class T> {
        const auto* upstream = grad.data<T>();
        const auto* saved = output.data<T>();
        auto* target = result.mutable_data<T>();
        const auto grad_step = grad.layout.strides[axis], output_step = output.layout.strides[axis];
        for (Index row = 0; row < rows; ++row) {
            const auto base = (row / inner) * width * inner + row % inner;
            const auto grad_base = grad.layout.index_offset(base), output_base = output.layout.index_offset(base);
            T total = 0;
            for (Index item = 0; item < width; ++item) {
                const T value = upstream[grad_base + item * grad_step];
                if constexpr (Logarithmic) total += value;
                else total += value * saved[output_base + item * output_step];
            }
            for (Index item = 0; item < width; ++item) {
                const auto index = base + item * inner;
                const T value = upstream[grad_base + item * grad_step];
                const T probability = saved[output_base + item * output_step];
                if constexpr (Logarithmic) target[index] = value - std::exp(probability) * total;
                else target[index] = probability * (value - total);
            }
        }
    });
    return result;
}

} // namespace

TensorData softmax(const TensorData& input, Index axis, bool logarithmic) {
    return logarithmic ? softmax_forward<true>(input, axis) : softmax_forward<false>(input, axis);
}
TensorData softmax_backward(const TensorData& grad, const TensorData& output, Index axis, bool logarithmic) {
    return logarithmic ? softmax_gradient<true>(grad, output, axis) : softmax_gradient<false>(grad, output, axis);
}

} // namespace dl::detail::cpu
