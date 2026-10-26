#include "core/layout.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace dl::detail {

Index numel(std::span<const Index> shape) {
    return std::accumulate(shape.begin(), shape.end(), Index{1}, std::multiplies<Index>{});
}

Index axis_index(Index axis, Index rank) { return axis < 0 ? axis + rank : axis; }

Shape insert_axis(std::span<const Index> values, Index axis, Index value) {
    Shape result(values.size() + 1);
    std::copy_n(values.begin(), axis, result.begin());
    result[axis] = value;
    std::copy(values.begin() + axis, values.end(), result.begin() + axis + 1);
    return result;
}

Shape resolve_shape(Shape shape, Index count) {
    Index inferred = -1;
    Index known = 1;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        if (shape[dim] == -1) {
            if (inferred >= 0) throw std::invalid_argument("reshape accepts one inferred dimension");
            inferred = static_cast<Index>(dim);
        } else {
            if (shape[dim] < 0) throw std::invalid_argument("reshape dimensions must be nonnegative or -1");
            if (shape[dim] && known > std::numeric_limits<Index>::max() / shape[dim])
                throw std::overflow_error("reshape element count exceeds Index range");
            known *= shape[dim];
        }
    }
    if (inferred >= 0) {
        if (known == 0 || count % known != 0)
            throw std::invalid_argument("reshape cannot infer an integral dimension");
        shape[static_cast<std::size_t>(inferred)] = count / known;
    } else if (known != count) {
        throw std::invalid_argument("reshape must preserve the element count");
    }
    return shape;
}

Shape broadcast_shape(std::span<const Index> a, std::span<const Index> b) {
    Shape shape(std::max(a.size(), b.size()), 1);
    for (std::size_t trailing = 0; trailing < shape.size(); ++trailing) {
        const Index left = trailing < a.size() ? a[a.size() - 1 - trailing] : 1;
        const Index right = trailing < b.size() ? b[b.size() - 1 - trailing] : 1;
        if (left != right && left != 1 && right != 1)
            throw std::invalid_argument("incompatible broadcast dimensions: " + std::to_string(left) + " and " + std::to_string(right));
        shape[shape.size() - 1 - trailing] = left == 1 ? right : left;
    }
    return shape;
}

Shape all_axes(Index rank) {
    Shape axes(static_cast<std::size_t>(rank));
    std::iota(axes.begin(), axes.end(), Index{0});
    return axes;
}

Shape normalize_axes(Shape axes, Index rank) {
    for (auto& axis : axes) {
        axis = axis_index(axis, rank);
    }
    std::sort(axes.begin(), axes.end());
    return axes;
}

Index Layout::numel() const { return detail::numel(sizes); }

bool is_contiguous(std::span<const Index> shape, std::span<const Index> strides) {
    if (numel(shape) == 0) {
        return true;
    }
    Index expected = 1;
    for (std::size_t dim = shape.size(); dim-- > 0;) {
        if (shape[dim] != 1) {
            if (strides[dim] != expected) {
                return false;
            }
            expected *= shape[dim];
        }
    }
    return true;
}

bool Layout::is_contiguous() const { return detail::is_contiguous(sizes, strides); }

Index Layout::index_offset(Index linear_index) const {
    Index result = 0;
    for (std::size_t dim = sizes.size(); dim-- > 0;) {
        result += (linear_index % sizes[dim]) * strides[dim];
        linear_index /= sizes[dim];
    }
    return result;
}

Layout contiguous_layout(Shape shape) {
    auto strides = contiguous_strides(shape);
    return {std::move(shape), std::move(strides), 0};
}

std::optional<Shape> reshape_strides(const Layout& source, std::span<const Index> shape) {
    if (source.numel() == 0 || source.is_contiguous()) {
        return contiguous_strides(shape);
    }

    Shape strides(shape.size());
    auto source_axis = source.sizes.size();
    auto target_axis = shape.size();
    while (source_axis > 0) {
        --source_axis;
        if (source.sizes[source_axis] == 1) {
            continue;
        }
        const Index stride = source.strides[source_axis];
        Index elements = source.sizes[source_axis];

        // 从内向外合并相邻的连续维，单例维不改变块中的逻辑元素顺序。
        while (source_axis > 0) {
            const auto previous = source_axis - 1;
            if (source.sizes[previous] != 1 && source.strides[previous] != stride * elements) {
                break;
            }
            elements *= source.sizes[previous];
            --source_axis;
        }

        // 当前块直接分配给目标末尾的若干维，目标维度只在同一存储块内拆分与合并。
        Index assigned = 1;
        while (target_axis > 0 && (assigned < elements || shape[target_axis - 1] == 1)) {
            --target_axis;
            strides[target_axis] = assigned * stride;
            assigned *= shape[target_axis];
        }
        if (assigned != elements) {
            return std::nullopt;
        }
    }
    return target_axis == 0 ? std::optional{std::move(strides)} : std::nullopt;
}

Shape broadcast_strides(const Layout& source, std::span<const Index> shape) {
    Shape strides(shape.size(), 0);
    const auto leading = shape.size() - source.sizes.size();
    for (std::size_t dim = 0; dim < source.sizes.size(); ++dim) {
        strides[leading + dim] = source.sizes[dim] == 1 ? 0 : source.strides[dim];
    }
    return strides;
}

} // namespace dl::detail
