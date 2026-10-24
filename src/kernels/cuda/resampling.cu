#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/resampling.h"
#include "kernels/cuda/common.cuh"

namespace dl::detail::cuda {
namespace {

template <bool Backward>
__global__ void adaptive_pool_kernel(const float* input, float* output, Index count,
                                     ResamplingGeometry geometry, IndexMap layout) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto region = geometry.window(index);
        float value = 0.0f;
        if constexpr (Backward) value = input[layout.offset(index)] / static_cast<float>(region.count());
        for (Index y = region.first_y; y < region.last_y; ++y) {
            for (Index x = region.first_x; x < region.last_x; ++x) {
                const auto pixel = geometry.offset(region.plane, y, x);
                if constexpr (Backward) atomicAdd(output + pixel, value);
                else value += input[layout.offset(pixel)];
            }
        }
        if constexpr (!Backward) output[index] = value / static_cast<float>(region.count());
    }
}

template <bool Backward>
__global__ void resize_kernel(const float* input, float* output, Index count, ResamplingGeometry geometry,
                              IndexMap layout, InterpolationMode mode, bool align_corners) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto point = geometry.point(index, mode, align_corners);
        float value = 0.0f;
        if constexpr (Backward) value = input[layout.offset(index)];
        for (int item = 0; item < point.count; ++item) {
            const float weight = static_cast<float>(point.weights[item]);
            if constexpr (Backward) atomicAdd(output + point.indices[item], value * weight);
            else value += input[layout.offset(point.indices[item])] * weight;
        }
        if constexpr (!Backward) output[index] = value;
    }
}

template <bool Backward, bool Adaptive>
void resample(const TensorData& source, TensorData& result,
              InterpolationMode mode = InterpolationMode::nearest, bool align_corners = false) {
    const auto geometry = resampling_geometry(Backward ? result.layout.sizes : source.layout.sizes,
        Backward ? source.layout.sizes : result.layout.sizes);
    const auto count = Backward ? source.layout.numel() : result.layout.numel();
    if (count != 0) {
        Execution execution(source);
        if constexpr (Adaptive) {
            adaptive_pool_kernel<Backward><<<grid_size(count), block_size, 0, execution.stream()>>>(
                source.data<float>(), result.mutable_data<float>(), count, geometry, IndexMap(source.layout));
        } else {
            resize_kernel<Backward><<<grid_size(count), block_size, 0, execution.stream()>>>(
                source.data<float>(), result.mutable_data<float>(), count, geometry, IndexMap(source.layout), mode, align_corners);
        }
        check(cudaGetLastError());
    }
}

} // namespace

TensorData adaptive_avg_pool2d(const TensorData& input, Size2 output_size) {
    auto result = empty_data(resampling_shape(input.layout.sizes, output_size), input.dtype, input.device());
    resample<false, true>(input, result);
    return result;
}
void adaptive_avg_pool2d_backward_into(TensorData& target, const TensorData& grad) {
    resample<true, true>(grad, target);
}
TensorData interpolate(const TensorData& input, Size2 output_size, InterpolationMode mode, bool align_corners) {
    auto result = empty_data(resampling_shape(input.layout.sizes, output_size), input.dtype, input.device());
    resample<false, false>(input, result, mode, align_corners);
    return result;
}
void interpolate_backward_into(TensorData& target, const TensorData& grad, InterpolationMode mode, bool align_corners) {
    resample<true, false>(grad, target, mode, align_corners);
}

} // namespace dl::detail::cuda
