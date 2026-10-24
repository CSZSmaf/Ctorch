#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

/// 周期坐标保留每个轴的边界，单次遍历将输出位置映射到输入的逻辑周期。
struct RepeatMap {
    int rank;
    Index sizes[max_rank]{}, periods[max_rank]{}, strides[max_rank]{};

    RepeatMap(const Shape& shape, const Layout& source) : rank(static_cast<int>(shape.size())) {
        const auto leading = shape.size() - source.sizes.size();
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            sizes[axis] = shape[axis];
            periods[axis] = axis < leading ? 1 : source.sizes[axis - leading];
            strides[axis] = axis < leading ? 0 : source.strides[axis - leading];
        }
    }

    __device__ Index offset(Index linear) const {
        Index result = 0;
        for (int axis = rank - 1; axis >= 0; --axis) {
            result += (linear % sizes[axis] % periods[axis]) * strides[axis];
            linear /= sizes[axis];
        }
        return result;
    }
};

template <class T>
__global__ void repeat_kernel(const T* input, T* output, Index count, RepeatMap source) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[index] = input[source.offset(index)];
    }
}

__global__ void repeat_backward_kernel(const float* grad, float* output, Index count,
                                       IndexMap source, RepeatMap target) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        atomicAdd(output + target.offset(index), grad[source.offset(index)]);
    }
}

template <class T>
__global__ void gather_kernel(const T* input, T* output, const Index* indices, Index count,
                               Index stride, IndexMap source, IndexMap selected) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[index] = input[source.offset(index) + indices[selected.offset(index)] * stride];
    }
}

__global__ void scatter_add_kernel(const float* values, float* output, const Index* indices, Index count,
                                    Index stride, IndexMap source, IndexMap target, IndexMap selected) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto position = target.offset(index) + indices[selected.offset(index)] * stride;
        // 重复索引并行累计到同一行，浮点加法的次序随线程调度确定。
        atomicAdd(output + position, values[source.offset(index)]);
    }
}

} // namespace

TensorData repeat(const TensorData& input, Shape shape) {
    auto result = empty_data(std::move(shape), input.dtype, input.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(input);
        cuda::visit_storage(input.dtype, [&]<class T> {
            repeat_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                input.data<T>(), result.mutable_data<T>(), count, RepeatMap(result.layout.sizes, input.layout));
        });
        check(cudaGetLastError());
    }
    return result;
}

void repeat_backward_into(TensorData& target, const TensorData& grad) {
    const auto count = grad.layout.numel();
    if (count != 0) {
        Execution execution(grad);
        repeat_backward_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            grad.data<float>(), target.mutable_data<float>(), count, IndexMap(grad.layout),
            RepeatMap(grad.layout.sizes, target.layout));
        check(cudaGetLastError());
    }
}

TensorData gather(const TensorData& input, Index axis, const TensorData& indices) {
    const auto& shape = indices.layout.sizes;
    auto result = empty_data(shape, input.dtype, input.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        auto source_strides = input.layout.strides;
        source_strides[axis] = 0;
        Execution execution(input);
        cuda::visit_storage(input.dtype, [&]<class T> {
            gather_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                input.data<T>(), result.mutable_data<T>(), indices.data<Index>(), count,
                input.layout.strides[axis], IndexMap(shape, source_strides), IndexMap(indices.layout));
        });
        check(cudaGetLastError());
    }
    return result;
}

void scatter_add_into(TensorData& target, Index axis, const TensorData& indices, const TensorData& source) {
    const auto count = indices.layout.numel();
    if (count != 0) {
        auto target_strides = target.layout.strides;
        target_strides[axis] = 0;
        Execution execution(target);
        scatter_add_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            source.data<float>(), target.mutable_data<float>(), indices.data<Index>(), count,
            target.layout.strides[axis], IndexMap(indices.layout.sizes, source.layout.strides),
            IndexMap(indices.layout.sizes, target_strides), IndexMap(indices.layout));
        check(cudaGetLastError());
    }
}

} // namespace dl::detail::cuda
