#pragma once

#include "core/spatial.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

/// 窗口沿通道、核位置、输出位置排列，越过图像边缘的项写为卷积填充值。
template <class T>
__global__ void im2col_kernel(const T* input, T* output, Index count,
                             SpatialGeometry g, IndexMap layout) {
    const auto positions = g.output_height * g.output_width;
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto row = index / positions;
        const auto column = index % positions;
        const auto x = (column % g.output_width) * g.stride_width - g.padding_width
            + (row % g.kernel_width) * g.dilation_width;
        const auto y = (column / g.output_width) * g.stride_height - g.padding_height
            + ((row / g.kernel_width) % g.kernel_height) * g.dilation_height;
        const auto plane = row / (g.kernel_height * g.kernel_width);
        output[index] = y >= 0 && y < g.height && x >= 0 && x < g.width
            ? input[layout.offset((plane * g.height + y) * g.width + x)] : storage_cast<T>(0.0f);
    }
}

__global__ void col2im_kernel(const float* columns, float* output, Index count,
                             SpatialGeometry g, IndexMap layout, bool accumulate) {
    const auto positions = g.output_height * g.output_width;
    const auto kernel_area = g.kernel_height * g.kernel_width;
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto x = index % g.width;
        const auto y = (index / g.width) % g.height;
        const auto plane = index / (g.height * g.width);
        float total = 0.0f;
        // 每个输入像素拥有一个线程，直接收集覆盖它的窗口，累计过程只写一次输出。
        for (Index ky = 0; ky < g.kernel_height; ++ky) {
            const auto oy = y + g.padding_height - ky * g.dilation_height;
            if (oy < 0 || oy % g.stride_height != 0 || oy / g.stride_height >= g.output_height) {
                continue;
            }
            for (Index kx = 0; kx < g.kernel_width; ++kx) {
                const auto ox = x + g.padding_width - kx * g.dilation_width;
                if (ox < 0 || ox % g.stride_width != 0 || ox / g.stride_width >= g.output_width) {
                    continue;
                }
                const auto row = plane * kernel_area + ky * g.kernel_width + kx;
                const auto column = (oy / g.stride_height) * g.output_width + ox / g.stride_width;
                total += columns[layout.offset(row * positions + column)];
            }
        }
        output[index] = accumulate ? output[index] + total : total;
    }
}

} // namespace
} // namespace dl::detail::cuda
