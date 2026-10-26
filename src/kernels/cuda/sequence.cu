#include "kernels/cuda/ops.h"
#include "backends/cuda_context.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"
#include "core/dispatch.h"

#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <math_constants.h>

namespace dl::detail::cuda {
namespace {

template <class T>
__global__ void scan_kernel(const T* input, T* output, Index rows, Index width, Index inner,
                           IndexMap layout, bool reverse) {
    using Scan = cub::BlockScan<T, block_size>;
    __shared__ typename Scan::TempStorage scratch;
    __shared__ T carry;
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = row / inner * width * inner + row % inner;
        if (threadIdx.x == 0) carry = T{};
        __syncthreads();
        for (Index start = 0; start < width; start += block_size) {
            const auto col = start + threadIdx.x;
            const auto position = base + (reverse ? width - 1 - col : col) * inner;
            T value = col < width ? input[layout.offset(position)] : T{}, prefix, total;
            Scan(scratch).InclusiveSum(value, prefix, total);
            if (col < width) output[position] = carry + prefix;
            __syncthreads();
            if (threadIdx.x == 0) carry += total;
            __syncthreads();
        }
    }
}

template <class T, class Key>
__global__ void pack_keys(const T* input, Key* keys, Index* indices, Index* offsets,
                          Index count, Index width, Index inner, IndexMap layout) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const auto row = i / width, col = i % width;
        auto value = static_cast<Key>(input[layout.offset((row / inner * width + col) * inner + row % inner)]);
        // 正号 NaN 排在正无穷之后；稳定基数排序保留同值与正负零的原索引顺序。
        if constexpr (std::is_floating_point_v<Key>) {
            if (value != value) value = copysign(static_cast<Key>(CUDART_NAN), Key{1});
            if (value == Key{}) value = Key{};
        }
        keys[i] = value;
        indices[i] = col;
        if (col == 0) offsets[row] = i;
        if (i == 0) offsets[count / width] = count;
    }
}

__global__ void arrange_indices(const Index* ordered, Index* output, Index count, Index width, Index inner) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const auto row = i / width, col = i % width;
        output[(row / inner * width + col) * inner + row % inner] = ordered[i] % width;
    }
}

template <class Key>
void radix_sort(Key* keys, Key* sorted, Index* indices, Index* ordered, Index* offsets,
                Index count, Index rows, bool descending, Device device, cudaStream_t stream) {
    std::size_t bytes = 0;
    auto launch = [&](void* workspace) {
        if (descending) return cub::DeviceSegmentedRadixSort::SortPairsDescending(workspace, bytes,
            keys, sorted, indices, ordered, count, rows, offsets, offsets + 1, 0, int(sizeof(Key) * 8), stream);
        return cub::DeviceSegmentedRadixSort::SortPairs(workspace, bytes,
            keys, sorted, indices, ordered, count, rows, offsets, offsets + 1, 0, int(sizeof(Key) * 8), stream);
    };
    check(launch(nullptr));
    auto workspace = empty_data({static_cast<Index>(bytes)}, DType::boolean, device);
    check(launch(workspace.mutable_data<void>()));
}

} // namespace

TensorData cumsum(const TensorData& input, Index axis, bool reverse) {
    auto result = empty_data(input.layout.sizes, input.dtype, input.device());
    if (input.layout.numel() == 0) return result;
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    Execution execution(input);
    visit_dtype(input.dtype, [&]<class T> {
        scan_kernel<<<row_grid_size(input.layout.numel() / width), block_size, 0, execution.stream()>>>(
            input.data<T>(), result.mutable_data<T>(), input.layout.numel() / width, width, inner,
            IndexMap(input.layout), reverse);
    });
    check(cudaGetLastError());
    return result;
}

TensorData argsort(const TensorData& input, Index axis, bool descending) {
    auto result = empty_data(input.layout.sizes, DType::i64, input.device());
    const auto count = input.layout.numel();
    if (count == 0) return result;
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    auto indices = empty_data({count}, DType::i64, input.device());
    auto ordered = empty_data({count}, DType::i64, input.device());
    auto offsets = empty_data({count / width + 1}, DType::i64, input.device());
    const auto key_dtype = input.dtype == DType::f64 || input.dtype == DType::i64 ? input.dtype : DType::f32;
    auto keys = empty_data({count}, key_dtype, input.device());
    auto sorted = empty_data({count}, key_dtype, input.device());
    Execution execution(input);
    // 把各轴行整理为连续段，CUB 在各段内独立排序，临时存储沿当前 stream 复用。
    visit_storage(input.dtype, [&]<class T> {
        using Key = std::conditional_t<std::is_same_v<T, double> || std::is_same_v<T, Index>, T, float>;
        pack_keys<<<grid_size(count), block_size, 0, execution.stream()>>>(input.data<T>(), keys.mutable_data<Key>(),
            indices.mutable_data<Index>(), offsets.mutable_data<Index>(), count, width, inner, IndexMap(input.layout));
        radix_sort(keys.mutable_data<Key>(), sorted.mutable_data<Key>(), indices.mutable_data<Index>(),
            ordered.mutable_data<Index>(), offsets.mutable_data<Index>(), count, count / width, descending,
            input.device(), execution.stream());
    });
    arrange_indices<<<grid_size(count), block_size, 0, execution.stream()>>>(ordered.data<Index>(), result.mutable_data<Index>(), count, width, inner);
    check(cudaGetLastError());
    return result;
}

} // namespace dl::detail::cuda
