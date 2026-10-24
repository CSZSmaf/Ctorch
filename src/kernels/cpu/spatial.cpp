#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/spatial.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dl::detail::cpu {
namespace {

Index pixel_offset(const Layout& layout, const SpatialGeometry& geometry, Index plane, Index y, Index x) {
    return (plane / geometry.channels) * layout.strides[0] + (plane % geometry.channels) * layout.strides[1]
        + y * layout.strides[2] + x * layout.strides[3];
}

// 每行窗口共享输入行和有效列区间，内层只沿列步长前进。
template <class Function>
void column_rows(const SpatialGeometry& g, Function function) {
    if (g.output_width == 0) return;
    Index offset = 0;
    for (Index plane = 0; plane < g.batch * g.channels; ++plane) {
        for (Index ky = 0; ky < g.kernel_height; ++ky) {
            for (Index kx = 0; kx < g.kernel_width; ++kx) {
                const auto x = kx * g.dilation_width - g.padding_width;
                const auto first = std::clamp((g.stride_width - 1 - x) / g.stride_width,
                                              Index{0}, g.output_width);
                const auto last = std::clamp((g.width + g.stride_width - 1 - x) / g.stride_width,
                                             first, g.output_width);
                auto y = ky * g.dilation_height - g.padding_height;
                for (Index row = 0; row < g.output_height; ++row, y += g.stride_height) {
                    function(plane, y, x, first, last, offset);
                    offset += g.output_width;
                }
            }
        }
    }
}

} // namespace

TensorData im2col(const TensorData& input, Size2 kernel, const Conv2dOptions& options) {
    const auto g = spatial_geometry(input.layout.sizes, kernel, options.stride, options.padding, options.dilation);
    auto result = empty_data(column_shape(g, options.groups), input.dtype);
    visit_storage(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        const auto step = g.stride_width * input.layout.strides[3];
        column_rows(g, [&](Index plane, Index y, Index x, Index first, Index last, Index offset) {
            auto* row = target + offset;
            if (y < 0 || y >= g.height) {
                std::fill_n(row, g.output_width, T{0});
                return;
            }
            std::fill_n(row, first, T{0});
            auto position = pixel_offset(input.layout, g, plane, y, x + first * g.stride_width);
            for (Index column = first; column < last; ++column, position += step)
                row[column] = source[position];
            std::fill(row + last, row + g.output_width, T{0});
        });
    });
    return result;
}

void col2im_into(TensorData& result, const TensorData& columns, Size2 kernel,
                 const Conv2dOptions& options, bool accumulate) {
    const auto g = spatial_geometry(result.layout.sizes, kernel, options.stride, options.padding, options.dilation);
    visit_floating(columns.dtype, [&]<class T> {
        const auto* source = columns.data<T>();
        auto* target = result.mutable_data<T>();
        if (!accumulate) std::fill_n(target, result.layout.numel(), T{0});
        const auto step = columns.layout.strides.back();
        column_rows(g, [&](Index plane, Index y, Index x, Index first, Index last, Index offset) {
            if (y < 0 || y >= g.height || first == last) return;
            auto position = columns.layout.index_offset(offset + first);
            auto pixel = (plane * g.height + y) * g.width + x + first * g.stride_width;
            for (Index column = first; column < last; ++column, position += step, pixel += g.stride_width)
                target[pixel] += source[position];
        });
    });
}

numeric::MaxPoolResult max_pool2d(const TensorData& input, const Pool2dOptions& options, bool save_indices) {
    const auto g = spatial_geometry(input.layout.sizes, options.kernel, options.stride, options.padding);
    auto result = empty_data(spatial_output(g), input.dtype);
    auto indices = save_indices ? std::optional{empty_data(result.layout.sizes, DType::i64)} : std::nullopt;
    auto* selected = indices ? indices->mutable_data<Index>() : nullptr;
    const auto count = result.layout.numel();
    visit_floating(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            const auto region = g.pool_window(index);
            T maximum = -std::numeric_limits<T>::infinity();
            Index best = -1;
            for (Index y = region.first_y; y < region.last_y; ++y) {
                for (Index x = region.first_x; x < region.last_x; ++x) {
                    const auto logical = (region.plane * g.height + y) * g.width + x;
                    const T value = source[pixel_offset(input.layout, g, region.plane, y, x)];
                    // NaN 随窗口结果传播，多个 NaN 由行优先最后一个位置承接梯度。
                    if (best < 0 || value > maximum || std::isnan(value)) {
                        maximum = value;
                        best = logical;
                    }
                }
            }
            target[index] = maximum;
            if (selected) selected[index] = best;
        }
    });
    return {std::move(result), std::move(indices)};
}

void max_pool2d_backward_into(TensorData& result, const TensorData& grad, const TensorData& indices) {
    const auto* selected = indices.data<Index>();
    const auto count = grad.layout.numel();
    visit_floating(grad.dtype, [&]<class T> {
        const auto* source = grad.data<T>();
        auto* target = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            target[selected[indices.layout.index_offset(index)]] += source[grad.layout.index_offset(index)];
        }
    });
}

TensorData avg_pool2d(const TensorData& input, const Pool2dOptions& options, bool count_include_pad) {
    const auto g = spatial_geometry(input.layout.sizes, options.kernel, options.stride, options.padding);
    auto result = empty_data(spatial_output(g), input.dtype);
    const auto count = result.layout.numel();
    visit_floating(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            const auto region = g.pool_window(index);
            T total = 0;
            for (Index y = region.first_y; y < region.last_y; ++y) {
                for (Index x = region.first_x; x < region.last_x; ++x) {
                    total += source[pixel_offset(input.layout, g, region.plane, y, x)];
                }
            }
            const auto divisor = count_include_pad ? g.kernel_height * g.kernel_width : region.count();
            target[index] = total / static_cast<T>(divisor);
        }
    });
    return result;
}

void avg_pool2d_backward_into(TensorData& result, const TensorData& grad, const Pool2dOptions& options,
                              bool count_include_pad) {
    const auto g = spatial_geometry(result.layout.sizes, options.kernel, options.stride, options.padding);
    const auto count = grad.layout.numel();
    visit_floating(grad.dtype, [&]<class T> {
        const auto* source = grad.data<T>();
        auto* target = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            const auto region = g.pool_window(index);
            const auto divisor = count_include_pad ? g.kernel_height * g.kernel_width : region.count();
            const T value = source[grad.layout.index_offset(index)] / static_cast<T>(divisor);
            for (Index y = region.first_y; y < region.last_y; ++y) {
                for (Index x = region.first_x; x < region.last_x; ++x) {
                    target[(region.plane * g.height + y) * g.width + x] += value;
                }
            }
        }
    });
}

} // namespace dl::detail::cpu
