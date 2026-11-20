#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/reduction.cuh"

namespace dl::detail::cuda {
namespace {

/// 候选由值和轴内位置组成；严格极值、NaN 优先和最小索引共同确定合并顺序。
template <class T> __device__ bool prefer(T value, Index index, T best, Index selected, bool maximum) {
    if (index < 0) return false;
    if (selected < 0) return true;
    const bool value_nan = value != value;
    const bool best_nan = best != best;
    if (value_nan != best_nan) return value_nan;
    return (value_nan || value == best) ? index < selected : (maximum ? value > best : value < best);
}

template <class T>
__global__ void arg_extreme_kernel(const T* input, Index* output, Index rows, Index width,
                                   Index inner, IndexMap layout, bool maximum) {
    __shared__ T values[block_size];
    __shared__ Index positions[block_size];
    const int lane = threadIdx.x;
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto base = (row / inner) * width * inner + row % inner;
        T best{};
        Index selected = -1;
        for (Index item = lane; item < width; item += blockDim.x) {
            const T value = input[layout.offset(base + item * inner)];
            if (prefer(value, item, best, selected, maximum)) {
                best = value;
                selected = item;
            }
        }
        values[lane] = best;
        positions[lane] = selected;
        __syncthreads();
        for (int distance = block_size / 2; distance > 0; distance /= 2) {
            if (lane < distance && prefer(values[lane + distance], positions[lane + distance],
                                          values[lane], positions[lane], maximum)) {
                values[lane] = values[lane + distance];
                positions[lane] = positions[lane + distance];
            }
            __syncthreads();
        }
        if (lane == 0) output[row] = positions[0];
        __syncthreads();
    }
}

template <bool Maximum>
void reduce_into(TensorData& result, const TensorData& input, const Shape& axes, bool accumulate) {
    const auto groups = result.layout.numel();
    if (groups == 0) return;
    Shape outer_shape, outer_strides, reduced_shape, reduced_strides;
    for (std::size_t axis = 0; axis < input.layout.sizes.size(); ++axis) {
        const bool reduced = std::binary_search(axes.begin(), axes.end(), static_cast<Index>(axis));
        (reduced ? reduced_shape : outer_shape).push_back(input.layout.sizes[axis]);
        (reduced ? reduced_strides : outer_strides).push_back(input.layout.strides[axis]);
    }
    const auto count = numel(reduced_shape);
    // 前导维归约可视为连续矩阵的列归约，也覆盖 Linear 偏置和广播梯度的求和。
    const bool leading = !axes.empty() && axes.back() == static_cast<Index>(axes.size()) - 1;
    const auto parts = std::max(Index{1}, (count + column_tile_rows - 1) / column_tile_rows);
    // 多分区路径需要足够的并行任务，以抵消合并结果时额外启动的开销。
    if (leading && groups >= warp_size && input.layout.is_contiguous()
        && (parts == 1 || parts * groups >= block_size * warp_size)) {
        TensorData partial = parts == 1 ? result : empty_data({parts, groups}, input.dtype, input.device());
        {
            Execution execution(input);
            reduce_columns_kernel<Maximum><<<grid_size(parts * groups), block_size, 0, execution.stream()>>>(
                input.data<float>(), partial.mutable_data<float>(), count, groups, parts, parts == 1 && accumulate);
            check(cudaGetLastError());
        }
        if (parts != 1) reduce_into<Maximum>(result, partial, {0}, accumulate);
        return;
    }
    const auto tiles = std::max(Index{1}, (count + reduction_tile - 1) / reduction_tile);
    TensorData partial = tiles == 1 ? result : empty_data({groups, tiles}, input.dtype, input.device());
    // 长归约分块生成局部结果，再按相同计算规则合并，使小输出也能使用多个 block。
    {
        Execution execution(input);
        reduce_kernel<Maximum><<<row_grid_size(groups * tiles), block_size, 0, execution.stream()>>>(
            input.data<float>(), partial.mutable_data<float>(), groups * tiles, count, tiles,
            IndexMap(outer_shape, outer_strides), IndexMap(reduced_shape, reduced_strides), tiles == 1 && accumulate);
        check(cudaGetLastError());
    }
    if (tiles != 1) reduce_into<Maximum>(result, partial, {1}, accumulate);
}

template <bool Maximum> TensorData reduce(const TensorData& input, const Shape& axes, KeepDim keep) {
    auto result = empty_data(numeric::reduced_shape(input.layout.sizes, axes, keep), input.dtype, input.device());
    reduce_into<Maximum>(result, input, axes, false);
    return result;
}

} // namespace

TensorData sum(const TensorData& input, const Shape& axes, KeepDim keep) {
    return reduce<false>(input, axes, keep);
}

void sum_into(TensorData& target, const TensorData& input, const Shape& axes, bool accumulate) {
    reduce_into<false>(target, input, axes, accumulate);
}

TensorData amax(const TensorData& input, const Shape& axes, KeepDim keep) {
    return reduce<true>(input, axes, keep);
}

TensorData arg_extreme(const TensorData& input, Index axis, KeepDim keep, bool maximum) {
    auto result = empty_data(numeric::reduced_shape(input.layout.sizes, {axis}, keep),
                             DType::i64, input.device());
    const auto rows = result.layout.numel();
    if (rows == 0) return result;
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    Execution execution(input);
    visit_dtype(input.dtype, [&]<class T> {
        arg_extreme_kernel<<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
            input.data<T>(), result.mutable_data<Index>(), rows, width, inner, IndexMap(input.layout), maximum);
    });
    check(cudaGetLastError());
    return result;
}

namespace {

template <bool Logarithmic> TensorData softmax_forward(const TensorData& input, Index axis) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype), input.device());
    if (input.layout.numel() == 0) {
        return result;
    }
    const auto width = input.layout.sizes[axis];
    const auto inner = numel(std::span(input.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    const auto rows = input.layout.numel() / width;
    Execution execution(input);
    visit_storage(input.dtype, [&]<class T> {
        softmax_kernel<Logarithmic><<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
            input.data<T>(), result.mutable_data<float>(), rows, width, inner, IndexMap(input.layout));
    });
    check(cudaGetLastError());
    return result;
}

// 每行只归约可见的 key。矩阵乘输出的连续存储依次保存指数和最终概率。
__global__ void attention_softmax_kernel(float* scores, Index rows, Index width, Index queries,
                                         Index first, float scale) {
    __shared__ float shared[block_size];
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        auto* values = scores + row * width;
        const auto end = first < 0 ? width : min(width, first + row % queries + 1);
        float maximum = -CUDART_INF_F;
        for (Index i = threadIdx.x; i < end; i += blockDim.x)
            maximum = reduce_pair<true>(maximum, values[i] * scale);
        maximum = block_reduce<true>(maximum, shared);
        float total = 0.0f;
        for (Index i = threadIdx.x; i < end; i += blockDim.x) {
            // 先舍入缩放后的分数，最大项减自身应恰好得到零。
            values[i] = expf(__fmul_rn(values[i], scale) - maximum);
            total += values[i];
        }
        total = block_reduce<false>(total, shared);
        for (Index i = threadIdx.x; i < width; i += blockDim.x)
            values[i] = i < end ? values[i] / total : 0.0f;
    }
}

template <bool Logarithmic>
TensorData softmax_gradient(const TensorData& grad, const TensorData& output, Index axis, float scale = 1.0f) {
    auto result = empty_data(output.layout.sizes, output.dtype, output.device());
    if (output.layout.numel() == 0) return result;
    const auto width = output.layout.sizes[axis];
    const auto inner = numel(std::span(output.layout.sizes).subspan(static_cast<std::size_t>(axis + 1)));
    const auto rows = output.layout.numel() / width;
    Execution execution(output);
    softmax_backward_kernel<Logarithmic><<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
        grad.data<float>(), output.data<float>(), result.mutable_data<float>(), rows, width, inner,
        IndexMap(grad.layout), IndexMap(output.layout), scale);
    check(cudaGetLastError());
    return result;
}

} // namespace

TensorData softmax(const TensorData& input, Index axis, bool logarithmic) {
    return logarithmic ? softmax_forward<true>(input, axis) : softmax_forward<false>(input, axis);
}
TensorData softmax_backward(const TensorData& grad, const TensorData& output, Index axis, bool logarithmic) {
    return logarithmic ? softmax_gradient<true>(grad, output, axis) : softmax_gradient<false>(grad, output, axis);
}

void attention_softmax_inplace(TensorData& scores, float scale, Index first) {
    if (scores.layout.numel() == 0) return;
    const auto& shape = scores.layout.sizes;
    const auto width = shape[3], rows = scores.layout.numel() / width;
    Execution execution(scores);
    attention_softmax_kernel<<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
        scores.mutable_data<float>(), rows, width, shape[2], first, scale);
    check(cudaGetLastError());
}

TensorData attention_softmax_backward(const TensorData& grad, const TensorData& weights, float scale) {
    return softmax_gradient<false>(grad, weights, 3, scale);
}

} // namespace dl::detail::cuda
