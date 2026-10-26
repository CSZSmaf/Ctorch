#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dl::detail::cpu {

using numeric::reduced_shape;
namespace {

IterationPlan<2> reduction_plan(const TensorData& input, const Shape& axes) {
    auto kept_shape = reduced_shape(input.layout.sizes, axes, KeepDim::yes);
    auto output_strides = contiguous_strides(kept_shape);
    for (auto axis : axes) {
        output_strides[axis] = 0;
    }
    return {input.layout.sizes, {input.layout.strides, std::move(output_strides)}};
}

template <bool Maximum>
void reduce_into(TensorData& result, const TensorData& input, const Shape& axes,
                 bool accumulate, double divisor = 1.0) {
    auto plan = reduction_plan(input, axes);
    visit_floating(input.dtype, [&]<class T> {
        using A = std::conditional_t<Maximum, T, double>;
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        const auto count = result.layout.numel();
        std::vector<A> buffer;
        A* totals;
        if constexpr (std::is_same_v<A, T>) totals = target;
        else {
            buffer.resize(static_cast<std::size_t>(count));
            totals = buffer.data();
            if (accumulate) std::copy_n(target, count, totals);
        }
        // 首次覆盖填入归约单位元，已有目标从此前汇合的值继续计算。
        if (!accumulate) std::fill_n(totals, count,
            Maximum ? -std::numeric_limits<A>::infinity() : A{0});
        auto combine = [](A accumulated, A value) {
            if constexpr (Maximum) {
                // NaN 是最大值归约的吸收值，后续有限值保持已产生的 NaN。
                return std::isnan(value) ? value : std::max(accumulated, value);
            } else {
                return accumulated + value;
            }
        };
        plan.for_each_row([&](Index, Index width, const auto& offsets, const auto& steps) {
            if (steps[1] == 0) {
                // 同一行归约到一个值，宽精度累加器驻留在局部变量中。
                A accumulated = totals[offsets[1]];
                for (Index column = 0; column < width; ++column)
                    accumulated = combine(accumulated, source[offsets[0] + column * steps[0]]);
                totals[offsets[1]] = accumulated;
            } else {
                for (Index column = 0; column < width; ++column) {
                    auto& accumulated = totals[offsets[1] + column * steps[1]];
                    accumulated = combine(accumulated, source[offsets[0] + column * steps[0]]);
                }
            }
        });
        if constexpr (!Maximum) {
            // 均值先在宽精度中除以元素数，最终写回时只舍入一次。
            for (Index i = 0; i < count; ++i) target[i] = static_cast<T>(totals[i] / divisor);
        }
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

TensorData mean(const TensorData& input, const Shape& axes, KeepDim keep) {
    auto result = empty_data(reduced_shape(input.layout.sizes, axes, keep), input.dtype);
    Index count = 1;
    for (auto axis : axes) count *= input.layout.sizes[axis];
    reduce_into<false>(result, input, axes, false, static_cast<double>(count));
    return result;
}

numeric::VarianceResult variance(const TensorData& input, const Shape& axes, double correction, KeepDim keep) {
    numeric::VarianceResult result{empty_data(reduced_shape(input.layout.sizes, axes, keep), input.dtype),
        empty_data(input.layout.sizes, input.dtype)};
    auto plan = reduction_plan(input, axes);
    Index count = 1;
    for (auto axis : axes) count *= input.layout.sizes[axis];
    const auto groups = result.output.layout.numel();
    std::vector<double> means(static_cast<std::size_t>(groups), 0.0);
    std::vector<double> squares(static_cast<std::size_t>(groups), 0.0);
    visit_floating(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* centered = result.centered.mutable_data<T>();
        auto* output = result.output.mutable_data<T>();
        plan.for_each([&](Index, const auto& offsets) { means[offsets[1]] += source[offsets[0]]; });
        for (auto& value : means) value /= static_cast<double>(count);
        // 均值与平方和保持 double；中心值直接供反向使用，省去逐元素平方张量。
        plan.for_each([&](Index index, const auto& offsets) {
            const double value = static_cast<double>(source[offsets[0]]) - means[offsets[1]];
            centered[index] = static_cast<T>(value);
            squares[offsets[1]] += value * value;
        });
        for (Index i = 0; i < groups; ++i)
            output[i] = static_cast<T>(squares[i] / (static_cast<double>(count) - correction));
    });
    return result;
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
            // 因果注意力的尾部掩码直接写入零概率；非有限行继续按完整宽度计算。
            Index active = width;
            if (std::isfinite(maximum)) {
                while (active > 0 && static_cast<A>(source[source_base + (active - 1) * step])
                    == -std::numeric_limits<A>::infinity()) --active;
            }
            A total = 0;
            for (Index item = 0; item < active; ++item) {
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
            for (Index item = 0; item < active; ++item) {
                auto& value = target[base + item * inner];
                if constexpr (Logarithmic) value -= log_total;
                else value /= total;
            }
            for (Index item = active; item < width; ++item)
                target[base + item * inner] = Logarithmic ? -std::numeric_limits<A>::infinity() : A{0};
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
