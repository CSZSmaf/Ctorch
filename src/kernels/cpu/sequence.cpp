#include "kernels/cpu/ops.h"
#include "core/dispatch.h"

#include <algorithm>
#include <numeric>

namespace dl::detail::cpu {

TensorData cumsum(const TensorData& input, Index axis, bool reverse) {
    auto result = empty_data(input.layout.sizes, input.dtype);
    if (input.layout.numel() == 0) return result;
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    visit_dtype(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        for (Index row = 0; row < input.layout.numel() / width; ++row) {
            const auto base = row / inner * width * inner + row % inner;
            // 每个前缀都按输出类型累积，整数输入保留整数表示。
            T total{};
            for (Index i = 0; i < width; ++i) {
                const auto position = base + (reverse ? width - 1 - i : i) * inner;
                total += source[input.layout.index_offset(position)];
                target[position] = total;
            }
        }
    });
    return result;
}

TensorData argsort(const TensorData& input, Index axis, bool descending) {
    auto result = empty_data(input.layout.sizes, DType::i64);
    if (input.layout.numel() == 0) return result;
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    std::vector<Index> order(static_cast<std::size_t>(width));
    visit_storage(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<Index>();
        for (Index row = 0; row < input.layout.numel() / width; ++row) {
            const auto base = row / inner * width * inner + row % inner;
            std::iota(order.begin(), order.end(), Index{0});
            std::sort(order.begin(), order.end(), [&](Index a, Index b) {
                const auto x = source[input.layout.index_offset(base + a * inner)];
                const auto y = source[input.layout.index_offset(base + b * inner)];
                const bool nan_x = x != x, nan_y = y != y;
                if (nan_x != nan_y) return descending ? nan_x : nan_y;
                if (nan_x || x == y) return a < b;
                return descending ? x > y : x < y;
            });
            for (Index i = 0; i < width; ++i) target[base + i * inner] = order[i];
        }
    });
    return result;
}

} // namespace dl::detail::cpu
