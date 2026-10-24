#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/spatial.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/convolution.cuh"

namespace dl::detail::cuda {
namespace {

__global__ void max_pool_kernel(const float* input, float* output, Index* selected, Index count,
                                 SpatialGeometry g, IndexMap layout) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto region = g.pool_window(index);
        float maximum = -CUDART_INF_F;
        Index best = -1;
        for (Index y = region.first_y; y < region.last_y; ++y) {
            for (Index x = region.first_x; x < region.last_x; ++x) {
                const auto logical = (region.plane * g.height + y) * g.width + x;
                const float value = input[layout.offset(logical)];
                // NaN 随窗口结果传播，多个 NaN 由行优先最后一个位置承接梯度。
                if (best < 0 || value > maximum || isnan(value)) {
                    maximum = value;
                    best = logical;
                }
            }
        }
        output[index] = maximum;
        if (selected) selected[index] = best;
    }
}

__global__ void max_pool_backward_kernel(const float* grad, const Index* selected, float* output,
                                          Index count, IndexMap source, IndexMap indices) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        atomicAdd(output + selected[indices.offset(index)], grad[source.offset(index)]);
    }
}

__global__ void avg_pool_kernel(const float* input, float* output, Index count,
                                 SpatialGeometry g, IndexMap layout, bool count_include_pad) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto region = g.pool_window(index);
        float total = 0.0f;
        for (Index y = region.first_y; y < region.last_y; ++y) {
            for (Index x = region.first_x; x < region.last_x; ++x) {
                total += input[layout.offset((region.plane * g.height + y) * g.width + x)];
            }
        }
        const auto divisor = count_include_pad ? g.kernel_height * g.kernel_width : region.count();
        output[index] = total / static_cast<float>(divisor);
    }
}

__global__ void avg_pool_backward_kernel(const float* grad, float* output, Index count,
                                          SpatialGeometry g, IndexMap layout, bool count_include_pad) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto region = g.pool_window(index);
        const auto divisor = count_include_pad ? g.kernel_height * g.kernel_width : region.count();
        const float value = grad[layout.offset(index)] / static_cast<float>(divisor);
        for (Index y = region.first_y; y < region.last_y; ++y) {
            for (Index x = region.first_x; x < region.last_x; ++x) {
                atomicAdd(output + (region.plane * g.height + y) * g.width + x, value);
            }
        }
    }
}

} // namespace

TensorData im2col(const TensorData& input, Size2 kernel, const Conv2dOptions& options) {
    const auto g = spatial_geometry(input.layout.sizes, kernel, options.stride, options.padding, options.dilation);
    auto result = empty_data(column_shape(g, options.groups), input.dtype, input.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(input);
        visit_storage(input.dtype, [&]<class T> {
            im2col_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                input.data<T>(), result.mutable_data<T>(), count, g, IndexMap(input.layout));
        });
        check(cudaGetLastError());
    }
    return result;
}

void col2im_into(TensorData& result, const TensorData& columns, Size2 kernel,
                 const Conv2dOptions& options, bool accumulate) {
    const auto g = spatial_geometry(result.layout.sizes, kernel, options.stride, options.padding, options.dilation);
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(columns);
        col2im_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            columns.data<float>(), result.mutable_data<float>(), count, g, IndexMap(columns.layout), accumulate);
        check(cudaGetLastError());
    }
}

numeric::MaxPoolResult max_pool2d(const TensorData& input, const Pool2dOptions& options, bool save_indices) {
    const auto g = spatial_geometry(input.layout.sizes, options.kernel, options.stride, options.padding);
    auto result = empty_data(spatial_output(g), input.dtype, input.device());
    auto indices = save_indices
        ? std::optional{empty_data(result.layout.sizes, DType::i64, input.device())} : std::nullopt;
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(input);
        max_pool_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            input.data<float>(), result.mutable_data<float>(), indices ? indices->mutable_data<Index>() : nullptr,
            count, g, IndexMap(input.layout));
        check(cudaGetLastError());
    }
    return {std::move(result), std::move(indices)};
}

void max_pool2d_backward_into(TensorData& result, const TensorData& grad, const TensorData& indices) {
    const auto count = grad.layout.numel();
    if (count != 0) {
        Execution execution(grad);
        max_pool_backward_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            grad.data<float>(), indices.data<Index>(), result.mutable_data<float>(), count,
            IndexMap(grad.layout), IndexMap(indices.layout));
        check(cudaGetLastError());
    }
}

TensorData avg_pool2d(const TensorData& input, const Pool2dOptions& options, bool count_include_pad) {
    const auto g = spatial_geometry(input.layout.sizes, options.kernel, options.stride, options.padding);
    auto result = empty_data(spatial_output(g), input.dtype, input.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(input);
        avg_pool_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            input.data<float>(), result.mutable_data<float>(), count, g, IndexMap(input.layout), count_include_pad);
        check(cudaGetLastError());
    }
    return result;
}

void avg_pool2d_backward_into(TensorData& result, const TensorData& grad, const Pool2dOptions& options,
                              bool count_include_pad) {
    const auto g = spatial_geometry(result.layout.sizes, options.kernel, options.stride, options.padding);
    const auto count = grad.layout.numel();
    if (count != 0) {
        Execution execution(grad);
        avg_pool_backward_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
            grad.data<float>(), result.mutable_data<float>(), count, g, IndexMap(grad.layout), count_include_pad);
        check(cudaGetLastError());
    }
}

} // namespace dl::detail::cuda
