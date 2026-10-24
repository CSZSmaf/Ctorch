#pragma once

#include "core/layout.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>

namespace dl::detail::cuda {

constexpr int block_size = 256;
constexpr int warp_size = 32;
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
    Index sizes[max_rank]{};
    Index strides[max_rank]{};

    IndexMap() = default;
    IndexMap(const Shape& shape, const Shape& steps) {
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
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
    }
    explicit IndexMap(const Layout& layout) : IndexMap(layout.sizes, layout.strides) {}

    __device__ Index offset(Index linear) const {
        if (rank == 1) {
            return linear * strides[0];
        }
        Index result = 0;
        for (int axis = rank - 1; axis >= 0; --axis) {
            result += (linear % sizes[axis]) * strides[axis];
            linear /= sizes[axis];
        }
        return result;
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

/// 一个 block 内的固定树形归约，所有线程得到同一个结果。
template <bool Maximum> __device__ float block_reduce(float value, float* shared) {
    const int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (int width = block_size / 2; width > 0; width /= 2) {
        if (lane < width) {
            shared[lane] = reduce_pair<Maximum>(shared[lane], shared[lane + width]);
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

} // namespace dl::detail::cuda
