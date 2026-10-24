#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "kernels/cuda/common.cuh"

#include <cmath>
#include <optional>

namespace dl::detail::cuda {
namespace {

constexpr int gradient_batch_capacity = 96;

struct GradientSegment {
    float* values;
    Index count;
    int begin, end;
};

// 描述表随 kernel 参数传入，梯度存储在整个提交过程内由调用方持有。
struct GradientBatch {
    GradientSegment segments[gradient_batch_capacity]{};
    int count = 0;
    int blocks = 0;
};

template <class Dense, class Strided>
void submit_gradients(std::span<TensorData> gradients, Dense dense, Strided strided) {
    GradientBatch batch;
    auto launch = [&] { dense(batch); batch = {}; };
    for (auto& gradient : gradients) {
        const auto count = gradient.layout.numel();
        if (count == 0) continue;
        if (!gradient.layout.is_contiguous()) {
            if (batch.count != 0) launch();
            strided(gradient);
            continue;
        }
        const int end = batch.blocks + grid_size(count);
        batch.segments[batch.count++] = {gradient.mutable_data<float>(), count, batch.blocks, end};
        batch.blocks = end;
        if (batch.count == gradient_batch_capacity) launch();
    }
    if (batch.count != 0) launch();
}

enum class GradientAction { scale, clip, unscale };

template <GradientAction Action>
__device__ int update_gradient(float& value, float factor) {
    if constexpr (Action == GradientAction::clip) {
        value = value < -factor ? -factor : value > factor ? factor : value;
    } else {
        value *= factor;
    }
    if constexpr (Action == GradientAction::unscale) return !isfinite(value);
    return 0;
}

template <GradientAction Action>
__device__ void finish_update(int invalid, float* found_inf) {
    if constexpr (Action == GradientAction::unscale) {
        // 还原后的有限性在 block 内合并，其他两种写回不产生标志或同步。
        const int any_invalid = __syncthreads_or(invalid);
        if (any_invalid && threadIdx.x == 0) atomicExch(found_inf, 1.0f);
    }
}

template <GradientAction Action>
__global__ void transform_batch_kernel(const GradientBatch batch, float value,
                                       const float* coefficient, float* found_inf) {
    const auto block = static_cast<int>(blockIdx.x);
    const auto segment = batch.segments[segment_index(batch, block)];
    const auto stride = static_cast<Index>(segment.end - segment.begin) * blockDim.x;
    const float factor = Action == GradientAction::scale ? *coefficient : value;
    int invalid = 0;
    for (Index i = static_cast<Index>(block - segment.begin) * blockDim.x + threadIdx.x;
         i < segment.count; i += stride) {
        invalid |= update_gradient<Action>(segment.values[i], factor);
    }
    finish_update<Action>(invalid, found_inf);
}

template <GradientAction Action>
__global__ void transform_strided_kernel(float* values, Index count, IndexMap layout,
                                         float value, const float* coefficient, float* found_inf) {
    const float factor = Action == GradientAction::scale ? *coefficient : value;
    int invalid = 0;
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<Index>(blockDim.x) * gridDim.x) {
        invalid |= update_gradient<Action>(values[layout.offset(i)], factor);
    }
    finish_update<Action>(invalid, found_inf);
}

template <GradientAction Action>
void transform_gradients(std::span<TensorData> gradients, float value,
                         const float* coefficient = nullptr, float* found_inf = nullptr) {
    Execution execution(gradients.front());
    submit_gradients(gradients,
        [&](const GradientBatch& batch) {
            transform_batch_kernel<Action><<<batch.blocks, block_size, 0, execution.stream()>>>(
                batch, value, coefficient, found_inf);
            check(cudaGetLastError());
        },
        [&](TensorData& gradient) {
            const auto count = gradient.layout.numel();
            transform_strided_kernel<Action><<<grid_size(count), block_size, 0, execution.stream()>>>(
                gradient.mutable_data<float>(), count, IndexMap(gradient.layout), value, coefficient, found_inf);
            check(cudaGetLastError());
        });
}

template <bool Maximum>
__device__ float norm_term(float value, float order, float scale) {
    const float absolute = fabsf(value);
    if constexpr (Maximum) return absolute;
    const float normalized = absolute / scale;
    return order == 1.0f ? normalized : order == 2.0f ? normalized * normalized : powf(normalized, order);
}

__device__ float norm_scale(const float* maximum) {
    const float scale = maximum ? *maximum : 1.0f;
    return scale == 0.0f ? 1.0f : scale;
}

template <bool Maximum>
__global__ void norm_batch_kernel(const GradientBatch batch, float order, const float* maximum, float* partial) {
    const auto block = static_cast<int>(blockIdx.x);
    const auto segment = batch.segments[segment_index(batch, block)];
    const auto stride = static_cast<Index>(segment.end - segment.begin) * blockDim.x;
    const float scale = norm_scale(maximum);
    float value = 0.0f;
    for (Index i = static_cast<Index>(block - segment.begin) * blockDim.x + threadIdx.x;
         i < segment.count; i += stride) {
        value = reduce_pair<Maximum>(value, norm_term<Maximum>(segment.values[i], order, scale));
    }
    __shared__ float shared[block_size];
    value = block_reduce<Maximum>(value, shared);
    if (threadIdx.x == 0) partial[block] = value;
}

template <bool Maximum>
__global__ void norm_strided_kernel(const float* values, Index count, IndexMap layout,
                                    float order, const float* maximum, float* partial) {
    float value = 0.0f;
    const float scale = norm_scale(maximum);
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<Index>(blockDim.x) * gridDim.x) {
        value = reduce_pair<Maximum>(value, norm_term<Maximum>(values[layout.offset(i)], order, scale));
    }
    __shared__ float shared[block_size];
    value = block_reduce<Maximum>(value, shared);
    if (threadIdx.x == 0) partial[blockIdx.x] = value;
}

template <bool Maximum>
__global__ void finish_norm_kernel(float* total, float* coefficient,
                                   float max_norm, float order, float inverse_order, const float* maximum) {
    float norm = *total;
    if constexpr (!Maximum) {
        if (order != 1.0f) norm = order == 2.0f ? sqrtf(norm) : powf(norm, inverse_order);
        const float scale = norm_scale(maximum);
        norm = isinf(scale) ? scale : scale * norm;
    }
    *total = norm;
    const float ratio = max_norm / (norm + 1e-6f);
    *coefficient = ratio < 0.0f ? 0.0f : ratio > 1.0f ? 1.0f : ratio;
}

template <bool Maximum>
TensorData reduce_norm(std::span<TensorData> gradients, TensorData& partial, cudaStream_t stream,
                       float order, const float* maximum = nullptr) {
    Index offset = 0;
    // 每个 block 写入独立槽，之后用已有的分块归约合并，无需浮点原子累计。
    submit_gradients(gradients,
        [&](const GradientBatch& batch) {
            norm_batch_kernel<Maximum><<<batch.blocks, block_size, 0, stream>>>(
                batch, order, maximum, partial.mutable_data<float>() + offset);
            check(cudaGetLastError());
            offset += batch.blocks;
        },
        [&](TensorData& gradient) {
            const auto count = gradient.layout.numel();
            const auto grid = grid_size(count);
            norm_strided_kernel<Maximum><<<grid, block_size, 0, stream>>>(gradient.data<float>(), count,
                IndexMap(gradient.layout), order, maximum, partial.mutable_data<float>() + offset);
            check(cudaGetLastError());
            offset += grid;
        });
    return Maximum ? amax(partial, {0}, KeepDim::no) : sum(partial, {0}, KeepDim::no);
}

template <bool Maximum>
TensorData norm_and_scale(std::span<TensorData> gradients, double max_norm, double order) {
    const auto device = gradients.front().device();
    Index blocks = 0;
    for (const auto& gradient : gradients) blocks += grid_size(gradient.layout.numel());
    if (blocks == 0) return full({}, 0.0, DType::f32, device);
    auto partial = empty_data({blocks}, DType::f32, device);
    Execution execution(partial);
    std::optional<TensorData> maximum;
    if constexpr (!Maximum) {
        if (order != 1.0) maximum = reduce_norm<true>(gradients, partial, execution.stream(), static_cast<float>(order));
    }
    const auto* scale = maximum ? maximum->data<float>() : nullptr;
    TensorData total = reduce_norm<Maximum>(gradients, partial, execution.stream(), static_cast<float>(order), scale);
    auto coefficient = empty_data({}, DType::f32, device);
    finish_norm_kernel<Maximum><<<1, 1, 0, execution.stream()>>>(total.mutable_data<float>(),
        coefficient.mutable_data<float>(), static_cast<float>(max_norm), static_cast<float>(order),
        static_cast<float>(1.0 / order), scale);
    check(cudaGetLastError());
    transform_gradients<GradientAction::scale>(gradients, 0.0f, coefficient.data<float>());
    return total;
}

} // namespace

TensorData unscale_gradients(std::span<TensorData> gradients, double inverse_scale) {
    auto flag = full({}, 0.0, DType::f32, gradients.front().device());
    transform_gradients<GradientAction::unscale>(gradients, static_cast<float>(inverse_scale),
        nullptr, flag.mutable_data<float>());
    return flag;
}

TensorData clip_grad_norm(std::span<TensorData> gradients, double max_norm, double order) {
    return std::isinf(order) ? norm_and_scale<true>(gradients, max_norm, order)
                           : norm_and_scale<false>(gradients, max_norm, order);
}

void clip_grad_value(std::span<TensorData> gradients, double bound) {
    transform_gradients<GradientAction::clip>(gradients, static_cast<float>(bound));
}

} // namespace dl::detail::cuda
