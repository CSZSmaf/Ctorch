#pragma once

#include "core/layout.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <limits>

namespace dl::detail::cuda {

constexpr int block_size = 256;
constexpr int warp_size = 32;
constexpr int block_warps = block_size / warp_size;
constexpr int max_rank = 16;

inline int grid_size(Index count) {
    return static_cast<int>(std::min((count + block_size - 1) / block_size, Index{4096}));
}
inline int row_grid_size(Index count) { return static_cast<int>(std::min(count, Index{4096})); }

/// 描述表按 block 区间排列，梯度处理与优化器共用同一个定位过程。
template <class Batch>
__device__ int segment_index(const Batch& batch, int block) {
    int first = 0, last = batch.count;
    while (first < last) {
        const auto middle = (first + last) / 2;
        if (block < batch.segments[middle].end) last = middle;
        else first = middle + 1;
    }
    return first;
}

/// 布局按值作为 kernel 参数传入，offset 已由 TensorData 的首元素地址吸收。
/// 至多 16 维的非负步长同时表示连续数据、转置、切片与广播。
struct IndexMap {
    int rank = 0;
    bool narrow = true;
    Index sizes[max_rank]{};
    Index strides[max_rank]{};

    IndexMap() = default;
    IndexMap(const Shape& shape, const Shape& steps) {
        Index count = 1;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            count *= shape[axis];
            if (shape[axis] == 1) {
                continue;
            }
            if (rank > 0 && strides[rank - 1] == steps[axis] * shape[axis]) {
                sizes[rank - 1] *= shape[axis];
                strides[rank - 1] = steps[axis];
            } else {
                sizes[rank] = shape[axis];
                strides[rank++] = steps[axis];
            }
        }
        narrow = count <= std::numeric_limits<std::uint32_t>::max();
    }
    explicit IndexMap(const Layout& layout) : IndexMap(layout.sizes, layout.strides) {}

    __device__ Index offset(Index linear) const {
        if (rank == 1) {
            return linear * strides[0];
        }
        return narrow ? offset_from(static_cast<std::uint32_t>(linear)) : offset_from(linear);
    }

    // 坐标除法按逻辑元素数选择整数宽度，存储偏移始终使用 64 位。
    template <class Coordinate>
    __device__ Index offset_from(Coordinate linear) const {
        Index result = 0;
        for (int axis = rank - 1; axis > 0; --axis) {
            const auto size = static_cast<Coordinate>(sizes[axis]);
            result += (linear % size) * strides[axis];
            linear /= size;
        }
        // 剩余下标已位于最外层轴范围内，直接使用其步长。
        return rank == 0 ? 0 : result + linear * strides[0];
    }
};

/// 最大值的合并保留 NaN；求和与最大值分别在编译期选定计算规则。
template <bool Maximum> __device__ float reduce_pair(float a, float b) {
    if constexpr (Maximum) {
        return isnan(a) ? a : isnan(b) ? b : fmaxf(a, b);
    } else {
        return a + b;
    }
}

/// warp 内通过寄存器交换合并，所有线程参与同一棵固定归约树。
template <bool Maximum> __device__ float warp_reduce(float value) {
    for (int width = warp_size / 2; width > 0; width /= 2)
        value = reduce_pair<Maximum>(value, __shfl_down_sync(0xffffffffu, value, width));
    return value;
}

/// 跨 warp 的部分按原树形顺序在寄存器中合并，最后由一个 warp 完成归约。
/// 调用方使用 block_size 个线程与共享槽；所有线程取得结果后再复用工作区。
template <bool Maximum> __device__ float block_reduce(float value, float* shared) {
    const int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    if (lane < warp_size) {
        float parts[block_warps];
        #pragma unroll
        for (int i = 0; i < block_warps; ++i) parts[i] = shared[lane + i * warp_size];
        #pragma unroll
        for (int width = block_warps / 2; width > 0; width /= 2) {
            #pragma unroll
            for (int i = 0; i < width; ++i) parts[i] = reduce_pair<Maximum>(parts[i], parts[i + width]);
        }
        value = warp_reduce<Maximum>(parts[0]);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    const float result = shared[0];
    __syncthreads();
    return result;
}

} // namespace dl::detail::cuda
