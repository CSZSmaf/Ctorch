#pragma once

#include "dl/types.h"

#include <algorithm>
#include <optional>
#include <span>

namespace dl::detail {

Index numel(std::span<const Index> shape);
bool is_contiguous(std::span<const Index> shape, std::span<const Index> strides);
Index axis_index(Index axis, Index rank);
/// 在已规范化的位置插入一个轴属性，供形状和步长共同使用。
Shape insert_axis(std::span<const Index> values, Index axis, Index value);

/// 行主序从最内层累计 stride，零长度轴沿用单例轴的布局间隔。
inline Shape contiguous_strides(std::span<const Index> shape) {
    Shape strides(shape.size());
    Index stride = 1;
    for (std::size_t dim = shape.size(); dim-- > 0;) {
        strides[dim] = stride;
        stride *= std::max(shape[dim], Index{1});
    }
    return strides;
}

Shape resolve_shape(Shape shape, Index count);
Shape broadcast_shape(std::span<const Index> a, std::span<const Index> b);
Shape all_axes(Index rank);
Shape normalize_axes(Shape axes, Index rank);

/// stride 和 offset 均以元素为单位，逻辑线性索引按行主序展开。
struct Layout {
    Shape sizes;
    Shape strides;
    Index offset = 0;

    Index numel() const;
    bool is_contiguous() const;
    Index index_offset(Index linear_index) const;
};

Layout contiguous_layout(Shape shape);
std::optional<Shape> reshape_strides(const Layout& source, std::span<const Index> shape);
Shape broadcast_strides(const Layout& source, std::span<const Index> shape);

} // namespace dl::detail
