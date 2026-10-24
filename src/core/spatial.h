#pragma once

#include "core/layout.h"
#include "dl/options.h"

#if defined(__CUDACC__)
#define CTORCH_SPATIAL_HD __host__ __device__
#else
#define CTORCH_SPATIAL_HD
#endif

namespace dl::detail {

/// 单个空间轴的有效窗口数量，供卷积、池化和固定计划的规格推导共用。
inline Index spatial_extent(Index input, Index kernel, Index stride, Index padding, Index dilation = 1) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

/// 池化窗口只列出真实输入区域，平均值的除数可选窗口面积或这一区域的元素数。
struct PoolWindow {
    Index plane, first_y, last_y, first_x, last_x;
    CTORCH_SPATIAL_HD Index count() const { return (last_y - first_y) * (last_x - first_x); }
};

/// 空间几何只保存整数，可作为 CPU 循环与 CUDA kernel 共用的值参数。
struct SpatialGeometry {
    Index batch, channels, height, width;
    Index kernel_height, kernel_width;
    Index stride_height, stride_width;
    Index padding_height, padding_width;
    Index dilation_height, dilation_width;
    Index output_height, output_width;

    CTORCH_SPATIAL_HD PoolWindow pool_window(Index index) const {
        const auto x = (index % output_width) * stride_width - padding_width;
        const auto y = ((index / output_width) % output_height) * stride_height - padding_height;
        return {index / (output_height * output_width), y > 0 ? y : 0,
            y + kernel_height < height ? y + kernel_height : height, x > 0 ? x : 0,
            x + kernel_width < width ? x + kernel_width : width};
    }
};

inline SpatialGeometry spatial_geometry(const Shape& shape, Size2 kernel, Size2 stride,
                                         Size2 padding, Size2 dilation = {1}) {
    const auto output_height = spatial_extent(shape[2], kernel.height, stride.height, padding.height, dilation.height);
    const auto output_width = spatial_extent(shape[3], kernel.width, stride.width, padding.width, dilation.width);
    return {shape[0], shape[1], shape[2], shape[3], kernel.height, kernel.width,
             stride.height, stride.width, padding.height, padding.width,
             dilation.height, dilation.width, output_height, output_width};
}

inline Shape spatial_output(const SpatialGeometry& geometry) {
    return {geometry.batch, geometry.channels, geometry.output_height, geometry.output_width};
}

inline Shape column_shape(const SpatialGeometry& geometry, Index groups) {
    return {geometry.batch, groups, geometry.channels / groups * geometry.kernel_height * geometry.kernel_width,
             geometry.output_height * geometry.output_width};
}

} // namespace dl::detail

#undef CTORCH_SPATIAL_HD
