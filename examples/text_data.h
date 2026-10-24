#pragma once

#include <dl/data.h>

#include <algorithm>
#include <string_view>

namespace examples {

/// 一个区间至少含两个字节，样本覆盖区间内的相邻字节预测，不跨越划分边界。
struct TextDataset {
    std::string_view text;
    dl::Index length;

    dl::Index size() const { return (static_cast<dl::Index>(text.size()) - 2) / length + 1; }

    dl::data::Batch operator()(std::span<const dl::Index> indices) const {
        const auto batch = static_cast<dl::Index>(indices.size());
        std::vector<dl::Index> input(static_cast<std::size_t>(batch * length), 0);
        std::vector<dl::Index> target(input.size(), -100);
        for (dl::Index row = 0; row < batch; ++row) {
            const auto begin = indices[static_cast<std::size_t>(row)] * length;
            const auto count = std::min(length, static_cast<dl::Index>(text.size()) - begin - 1);
            for (dl::Index column = 0; column < count; ++column) {
                const auto destination = static_cast<std::size_t>(row * length + column);
                input[destination] = static_cast<unsigned char>(text[static_cast<std::size_t>(begin + column)]);
                target[destination] = static_cast<unsigned char>(text[static_cast<std::size_t>(begin + column + 1)]);
            }
        }
        return {dl::from_span<dl::Index>(input, {batch, length}),
                dl::from_span<dl::Index>(target, {batch, length})};
    }
};

} // namespace examples
