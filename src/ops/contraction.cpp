#include "dl/ops.h"
#include "core/layout.h"

#include <algorithm>
#include <stdexcept>

namespace dl {

Tensor tensordot(const Tensor& a, const Tensor& b, Shape axes_a, Shape axes_b) {
    auto axes = [](Shape values, Index rank) {
        Shape seen;
        for (auto& axis : values) {
            axis = detail::axis_index(axis, rank);
            if (axis < 0 || axis >= rank || std::find(seen.begin(), seen.end(), axis) != seen.end())
                throw std::invalid_argument("tensordot axes must be distinct and in range");
            seen.push_back(axis);
        }
        return values;
    };
    axes_a = axes(std::move(axes_a), a.ndim());
    axes_b = axes(std::move(axes_b), b.ndim());
    if (axes_a.size() != axes_b.size()) throw std::invalid_argument("tensordot axis counts differ");
    Index inner = 1;
    for (std::size_t i = 0; i < axes_a.size(); ++i) {
        if (a.size(axes_a[i]) != b.size(axes_b[i])) throw std::invalid_argument("tensordot contracted dimensions differ");
        inner *= a.size(axes_a[i]);
    }
    auto remaining = [](Index rank, const Shape& contracted) {
        Shape result;
        for (Index i = 0; i < rank; ++i)
            if (std::find(contracted.begin(), contracted.end(), i) == contracted.end()) result.push_back(i);
        return result;
    };
    auto left = remaining(a.ndim(), axes_a), right = remaining(b.ndim(), axes_b);
    Shape output;
    Index rows = 1, columns = 1;
    for (auto axis : left) { output.push_back(a.size(axis)); rows *= a.size(axis); }
    for (auto axis : right) { output.push_back(b.size(axis)); columns *= b.size(axis); }
    left.insert(left.end(), axes_a.begin(), axes_a.end());
    axes_b.insert(axes_b.end(), right.begin(), right.end());
    return matmul(a.permute(left).reshape({rows, inner}), b.permute(axes_b).reshape({inner, columns})).reshape(output);
}

Tensor rotary_embedding(const Tensor& input, const Tensor& cosine, const Tensor& sine) {
    const auto dimensions = input.size(-1);
    if (dimensions == 0 || dimensions % 2 != 0) throw std::invalid_argument("rotary embedding requires a positive even dimension");
    auto even = input.slice(-1, 0, dimensions, 2), odd = input.slice(-1, 1, dimensions, 2);
    return stack({even * cosine - odd * sine, even * sine + odd * cosine}, -1).reshape(input.shape());
}

Tensor grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                std::optional<Tensor> keep_mask) {
    if (query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4)
        throw std::invalid_argument("grouped-query attention expects [batch,heads,length,features]");
    const auto batch = query.size(0), heads = query.size(1), kv_heads = key.size(1);
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || value.size(1) != kv_heads)
        throw std::invalid_argument("query heads must be divisible by matching key/value heads");
    if (key.size(0) != batch || value.size(0) != batch || query.size(3) != key.size(3) || key.size(2) != value.size(2))
        throw std::invalid_argument("attention batch, key dimension and key/value lengths must match");
    const auto groups = heads / kv_heads, length = query.size(2), source = key.size(2);
    auto q = query.reshape({batch, kv_heads, groups, length, query.size(3)});
    if (keep_mask && keep_mask->ndim() > 2)
        keep_mask = keep_mask->expand({batch, heads, length, source}).reshape({batch, kv_heads, groups, length, source});
    return scaled_dot_product_attention(q, key.unsqueeze(2), value.unsqueeze(2), std::move(keep_mask))
        .reshape({batch, heads, length, value.size(3)});
}

} // namespace dl
