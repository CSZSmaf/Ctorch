#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

template <class T>
__global__ void cross_entropy_kernel(const T* logits, const Index* labels, float* statistics,
    float* losses, Index rows, Index classes, Index inner, IndexMap input,
    IndexMap targets, bool has_ignore, Index ignore_index) {
    __shared__ float shared[block_size];
    for (Index row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto target = labels[targets.offset(row)];
        if (has_ignore && target == ignore_index) {
            if (threadIdx.x == 0) {
                losses[row] = 0;
                if (statistics) { statistics[row] = 0; statistics[rows + row] = 1; }
            }
            continue;
        }
        const auto base = (row / inner) * classes * inner + row % inner;
        float maximum = -CUDART_INF_F;
        for (Index column = threadIdx.x; column < classes; column += blockDim.x) {
            maximum = reduce_pair<true>(maximum,
                storage_cast<float>(logits[input.offset(base + column * inner)]));
        }
        maximum = block_reduce<true>(maximum, shared);
        float total = 0.0f;
        for (Index column = threadIdx.x; column < classes; column += blockDim.x) {
            total += expf(storage_cast<float>(logits[input.offset(base + column * inner)]) - maximum);
        }
        total = block_reduce<false>(total, shared);
        if (threadIdx.x == 0) {
            if (statistics) { statistics[row] = maximum; statistics[rows + row] = total; }
            // 最大值与目标 logit 先相减，损失保留平移后的对数归约项。
            losses[row] = (maximum - storage_cast<float>(logits[input.offset(base + target * inner)])) + logf(total);
        }
    }
}

/// 第一层读取逐项损失与标签，第二层合并固定数量的局部和；有效项数始终留在设备上。
template <bool Partial>
__global__ void loss_reduce_kernel(const float* input, const Index* labels, float* partial,
    float* loss, float* statistics, Index count, Index rows, IndexMap targets,
    bool has_ignore, Index ignore_index, bool mean) {
    __shared__ float shared[block_size];
    float total = 0, valid = 0;
    for (Index item = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < count; item += static_cast<Index>(blockDim.x) * gridDim.x) {
        total += input[item];
        if (mean) {
            if constexpr (Partial) valid += input[count + item];
            else valid += (!has_ignore || labels[targets.offset(item)] != ignore_index) ? 1.0f : 0.0f;
        }
    }
    total = block_reduce<false>(total, shared);
    if (mean) valid = block_reduce<false>(valid, shared);
    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            *loss = mean ? total / valid : total;
            if (statistics && mean) statistics[2 * rows] = valid;
        } else {
            partial[blockIdx.x] = total;
            partial[gridDim.x + blockIdx.x] = valid;
        }
    }
}

template <class T>
__global__ void cross_entropy_backward_kernel(const T* logits, const Index* labels,
    const float* statistics, const float* grad, float* output, Index count,
    Index rows, Index classes, Index inner, IndexMap input, IndexMap targets, IndexMap seeds,
    LossReduction reduction, bool has_ignore, Index ignore_index, bool accumulate) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto row = (index / (classes * inner)) * inner + index % inner;
        const auto column = (index / inner) % classes;
        const auto label = labels[targets.offset(row)];
        if (has_ignore && label == ignore_index) {
            if (!accumulate) output[index] = 0;
            continue;
        }
        float factor = grad[seeds.offset(reduction == LossReduction::none ? row : 0)];
        if (reduction == LossReduction::mean) factor /= statistics[2 * rows];
        const float probability = expf(storage_cast<float>(logits[input.offset(index)]) - statistics[row])
            / statistics[rows + row];
        const float value = (probability - (column == label ? 1.0f : 0.0f)) * factor;
        if (accumulate) output[index] += value;
        else output[index] = value;
    }
}

} // namespace

void cross_entropy(const TensorData& logits, const TensorData& labels, LossReduction reduction,
    std::optional<Index> ignore_index, CrossEntropyResult& result) {
    const std::size_t axis = logits.layout.sizes.size() == 1 ? 0 : 1;
    const auto classes = logits.layout.sizes[axis];
    const auto inner = numel(std::span(logits.layout.sizes).subspan(axis + 1));
    const auto rows = labels.layout.numel();
    auto losses = reduction == LossReduction::none ? result.loss
        : empty_data(labels.layout.sizes, result.loss.dtype, logits.device());
    // 小批次直接归约到标量，较长序列用有界分区并行归约，再合并两个局部统计量。
    const auto partitions = std::max(Index{1}, std::min(Index{128}, (rows + 4095) / 4096));
    std::optional<TensorData> partial;
    if (reduction != LossReduction::none && partitions > 1)
        partial = empty_data({2, partitions}, result.loss.dtype, logits.device());
    auto* statistics = result.statistics ? result.statistics->mutable_data<float>() : nullptr;
    Execution execution(logits);
    if (rows != 0) {
        visit_storage(logits.dtype, [&]<class T> {
            cross_entropy_kernel<<<row_grid_size(rows), block_size, 0, execution.stream()>>>(
                logits.data<T>(), labels.data<Index>(), statistics, losses.mutable_data<float>(), rows, classes, inner,
                IndexMap(logits.layout), IndexMap(labels.layout), ignore_index.has_value(), ignore_index.value_or(0));
        });
        check(cudaGetLastError());
    }
    if (reduction != LossReduction::none) {
        loss_reduce_kernel<false><<<static_cast<unsigned>(partitions), block_size, 0, execution.stream()>>>(
            losses.data<float>(), labels.data<Index>(), partial ? partial->mutable_data<float>() : nullptr,
            result.loss.mutable_data<float>(), statistics, rows, rows, IndexMap(labels.layout),
            ignore_index.has_value(), ignore_index.value_or(0), reduction == LossReduction::mean);
        check(cudaGetLastError());
        if (partial) {
            loss_reduce_kernel<true><<<1, block_size, 0, execution.stream()>>>(
                partial->data<float>(), nullptr, nullptr, result.loss.mutable_data<float>(), statistics,
                partitions, rows, IndexMap{}, false, 0, reduction == LossReduction::mean);
            check(cudaGetLastError());
        }
    }
}

void cross_entropy_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, const TensorData& statistics,
    LossReduction reduction, std::optional<Index> ignore_index, bool accumulate) {
    const auto count = target.layout.numel();
    if (count == 0) return;
    const std::size_t axis = logits.layout.sizes.size() == 1 ? 0 : 1;
    const auto classes = logits.layout.sizes[axis];
    const auto inner = numel(std::span(logits.layout.sizes).subspan(axis + 1));
    Execution execution(logits);
    visit_storage(logits.dtype, [&]<class T> {
        cross_entropy_backward_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            logits.data<T>(), labels.data<Index>(), statistics.data<float>(), grad.data<float>(), target.mutable_data<float>(),
            count, labels.layout.numel(), classes, inner, IndexMap(logits.layout), IndexMap(labels.layout), IndexMap(grad.layout),
            reduction, ignore_index.has_value(), ignore_index.value_or(0), accumulate);
    });
    check(cudaGetLastError());
}

} // namespace dl::detail::cuda
