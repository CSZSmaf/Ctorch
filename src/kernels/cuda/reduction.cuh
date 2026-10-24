#pragma once

#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

constexpr Index reduction_tile = 4096;

/// 每个 block 产生一个分块结果，tiles 决定同一逻辑输出使用的分块数量。
template <bool Maximum>
__global__ void reduce_kernel(const float* input, float* output, Index groups, Index count, Index tiles,
                             IndexMap outer, IndexMap reduced, bool accumulate) {
    __shared__ float shared[block_size];
    for (Index group = blockIdx.x; group < groups; group += gridDim.x) {
        const auto base = outer.offset(group / tiles);
        const auto begin = (group % tiles) * reduction_tile;
        const auto end = count < begin + reduction_tile ? count : begin + reduction_tile;
        float value = Maximum ? -CUDART_INF_F : 0.0f;
        for (Index item = begin + threadIdx.x; item < end; item += blockDim.x) {
            value = reduce_pair<Maximum>(value, input[base + reduced.offset(item)]);
        }
        value = block_reduce<Maximum>(value, shared);
        if (threadIdx.x == 0) {
            if (accumulate) value = reduce_pair<Maximum>(output[group], value);
            output[group] = value;
        }
    }
}

/// 一行依次归约最大值与指数和，概率与对数概率按编译期选择直接写入结果。
template <bool Logarithmic = false, class T>
__global__ void softmax_kernel(const T* input, float* output, Index rows, Index width,
                              Index inner, IndexMap layout) {
    __shared__ float shared[block_size];
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = (row / inner) * width * inner + row % inner;
        float maximum = -CUDART_INF_F;
        for (Index item = threadIdx.x; item < width; item += blockDim.x) {
            maximum = reduce_pair<true>(maximum, storage_cast<float>(input[layout.offset(base + item * inner)]));
        }
        maximum = block_reduce<true>(maximum, shared);
        float total = 0.0f;
        for (Index item = threadIdx.x; item < width; item += blockDim.x) {
            total += expf(storage_cast<float>(input[layout.offset(base + item * inner)]) - maximum);
        }
        total = block_reduce<false>(total, shared);
        float log_total = 0.0f;
        if constexpr (Logarithmic) log_total = logf(total);
        for (Index item = threadIdx.x; item < width; item += blockDim.x) {
            const auto index = base + item * inner;
            const float shifted = storage_cast<float>(input[layout.offset(index)]) - maximum;
            if constexpr (Logarithmic) output[index] = shifted - log_total;
            else output[index] = expf(shifted) / total;
        }
    }
}

/// softmax 用 sum(g*y)，log_softmax 用 sum(g)，行统计和 VJP 共用一次启动。
template <bool Logarithmic>
__global__ void softmax_backward_kernel(const float* grad, const float* saved, float* output,
    Index rows, Index width, Index inner, IndexMap grad_layout, IndexMap saved_layout) {
    __shared__ float shared[block_size];
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = (row / inner) * width * inner + row % inner;
        float total = 0.0f;
        for (Index item = threadIdx.x; item < width; item += blockDim.x) {
            const auto index = base + item * inner;
            const float value = grad[grad_layout.offset(index)];
            if constexpr (Logarithmic) total += value;
            else total += value * saved[saved_layout.offset(index)];
        }
        total = block_reduce<false>(total, shared);
        for (Index item = threadIdx.x; item < width; item += blockDim.x) {
            const auto index = base + item * inner;
            const float value = grad[grad_layout.offset(index)];
            const float probability = saved[saved_layout.offset(index)];
            if constexpr (Logarithmic) output[index] = value - expf(probability) * total;
            else output[index] = probability * (value - total);
        }
    }
}

} // namespace
} // namespace dl::detail::cuda
