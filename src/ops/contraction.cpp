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
            if (std::find(seen.begin(), seen.end(), axis) != seen.end())
                throw std::invalid_argument("tensordot axes must be distinct");
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

} // namespace dl
