#include "kernels/cpu/ops.h"

#include "core/dispatch.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dl::detail::cpu {

void cross_entropy(const TensorData& logits, const TensorData& labels, LossReduction reduction,
    std::optional<Index> ignore_index, CrossEntropyResult& result) {
    const std::size_t axis = logits.layout.sizes.size() == 1 ? 0 : 1;
    const auto classes = logits.layout.sizes[axis], class_stride = logits.layout.strides[axis];
    const auto inner = numel(std::span(logits.layout.sizes).subspan(axis + 1));
    const auto rows = labels.layout.numel();
    const auto* targets = labels.data<Index>();
    visit_storage(logits.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        const auto* source = logits.data<T>();
        auto* losses = result.loss.mutable_data<A>();
        auto* statistics = result.statistics ? result.statistics->mutable_data<A>() : nullptr;
        A total_loss = 0;
        Index valid = 0;
        for (Index row = 0; row < rows; ++row) {
            const auto target = targets[labels.layout.index_offset(row)];
            if (ignore_index && target == *ignore_index) {
                if (reduction == LossReduction::none) losses[row] = 0;
                if (statistics) { statistics[row] = 0; statistics[rows + row] = 1; }
                continue;
            }
            // 移除类别轴后的逻辑行顺序与 labels 一致，行内直接沿原类别步长访问。
            const auto base = logits.layout.index_offset((row / inner) * classes * inner + row % inner);
            A maximum = -std::numeric_limits<A>::infinity();
            for (Index column = 0; column < classes; ++column) {
                const A value = static_cast<A>(source[base + column * class_stride]);
                maximum = std::isnan(value) ? value : std::max(maximum, value);
            }
            A normalizer = 0;
            for (Index column = 0; column < classes; ++column) {
                normalizer += std::exp(static_cast<A>(source[base + column * class_stride]) - maximum);
            }
            if (statistics) {
                statistics[row] = maximum;
                statistics[rows + row] = normalizer;
            }
            // 先计算最大值与目标 logit 的差，再加入平移后的对数归约。
            const A loss = (maximum - static_cast<A>(source[base + target * class_stride])) + std::log(normalizer);
            if (reduction == LossReduction::none) losses[row] = loss;
            else total_loss += loss;
            ++valid;
        }
        if (statistics) statistics[2 * rows] = static_cast<A>(valid);
        if (reduction != LossReduction::none) {
            *losses = reduction == LossReduction::mean ? total_loss / static_cast<A>(valid) : total_loss;
        }
    });
}

void cross_entropy_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, const TensorData& statistics,
    LossReduction reduction, std::optional<Index> ignore_index, bool accumulate) {
    const std::size_t axis = logits.layout.sizes.size() == 1 ? 0 : 1;
    const auto classes = logits.layout.sizes[axis], class_stride = logits.layout.strides[axis];
    const auto inner = numel(std::span(logits.layout.sizes).subspan(axis + 1));
    const auto rows = labels.layout.numel();
    const auto* targets = labels.data<Index>();
    visit_storage(logits.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        const auto* source = logits.data<T>();
        const auto* saved = statistics.data<A>();
        const auto* upstream = grad.data<A>();
        auto* values = target.mutable_data<A>();
        for (Index row = 0; row < rows; ++row) {
            const auto label = targets[labels.layout.index_offset(row)];
            const auto base = (row / inner) * classes * inner + row % inner;
            if (ignore_index && label == *ignore_index) {
                // 忽略项对已有目标没有贡献；首次写入时对应位置取零。
                if (!accumulate) {
                    for (Index column = 0; column < classes; ++column) values[base + column * inner] = 0;
                }
                continue;
            }
            const auto input_base = logits.layout.index_offset(base);
            A factor = upstream[reduction == LossReduction::none ? grad.layout.index_offset(row) : 0];
            if (reduction == LossReduction::mean) factor /= saved[2 * rows];
            for (Index column = 0; column < classes; ++column) {
                const A probability = std::exp(static_cast<A>(source[input_base + column * class_stride])
                    - saved[row]) / saved[rows + row];
                const A value = (probability - (column == label ? A{1} : A{0})) * factor;
                auto& output = values[base + column * inner];
                if (accumulate) output += value;
                else output = value;
            }
        }
    });
}

} // namespace dl::detail::cpu
