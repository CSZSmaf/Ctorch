#include "kernels/cuda/ops.h"
#include "backends/cuda_context.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

struct Channels {
    Index channels, spatial = 1, count, parts;
    explicit Channels(const Layout& layout) : channels(layout.sizes[1]) {
        for (std::size_t i = 2; i < layout.sizes.size(); ++i) spatial *= layout.sizes[i];
        count = layout.sizes[0] * spatial;
        parts = std::max(Index{1}, std::min((count + 1023) / 1024, Index{128}));
    }
    __device__ Index index(Index channel, Index sample) const {
        return (sample / spatial * channels + channel) * spatial + sample % spatial;
    }
};

/// 每个分区保存均值与中心平方和，合并时再加入分区均值之间的方差。
template <class T>
__global__ void batch_moments_kernel(const T* input, float* partials, Channels shape, IndexMap layout) {
    __shared__ float shared[block_size];
    const auto tasks = shape.channels * shape.parts;
    for (Index task = blockIdx.x; task < tasks; task += gridDim.x) {
        const auto channel = task % shape.channels, part = task / shape.channels;
        const auto begin = part * shape.count / shape.parts, end = (part + 1) * shape.count / shape.parts;
        float sum = 0.0f;
        for (Index i = begin + threadIdx.x; i < end; i += blockDim.x)
            sum += storage_cast<float>(input[layout.offset(shape.index(channel, i))]);
        const float mean = block_reduce<false>(sum, shared) / static_cast<float>(end - begin);
        float squares = 0.0f;
        for (Index i = begin + threadIdx.x; i < end; i += blockDim.x) {
            const float centered = storage_cast<float>(input[layout.offset(shape.index(channel, i))]) - mean;
            squares += centered * centered;
        }
        squares = block_reduce<false>(squares, shared);
        if (threadIdx.x == 0) {
            partials[task] = mean;
            partials[tasks + task] = squares;
        }
    }
}

__global__ void batch_statistics_kernel(const float* partials, float* statistics,
    float* running_mean, float* running_variance, Channels shape, float epsilon,
    float decay, float mean_weight, float variance_weight, IndexMap mean_layout, IndexMap variance_layout) {
    const auto tasks = shape.channels * shape.parts;
    for (Index c = blockIdx.x * blockDim.x + threadIdx.x; c < shape.channels; c += gridDim.x * blockDim.x) {
        float mean = 0.0f, variance = 0.0f;
        if (partials) {
            for (Index part = 0; part < shape.parts; ++part) {
                const auto size = (part + 1) * shape.count / shape.parts - part * shape.count / shape.parts;
                mean += partials[part * shape.channels + c] * static_cast<float>(size);
            }
            mean /= static_cast<float>(shape.count);
            for (Index part = 0; part < shape.parts; ++part) {
                const auto size = (part + 1) * shape.count / shape.parts - part * shape.count / shape.parts;
                const float delta = partials[part * shape.channels + c] - mean;
                variance += partials[tasks + part * shape.channels + c] + delta * delta * static_cast<float>(size);
            }
            variance /= static_cast<float>(shape.count);
            if (running_mean) {
                const auto m = mean_layout.offset(c), v = variance_layout.offset(c);
                running_mean[m] = decay * running_mean[m] + mean_weight * mean;
                running_variance[v] = decay * running_variance[v] + variance_weight * variance;
            }
        } else {
            mean = running_mean[mean_layout.offset(c)];
            variance = running_variance[variance_layout.offset(c)];
        }
        statistics[c] = mean;
        statistics[shape.channels + c] = rsqrtf(variance + epsilon);
    }
}

template <class T>
__global__ void batch_affine_kernel(const T* input, const float* statistics, const float* weight,
    const float* bias, float* output, Channels shape, IndexMap layout, IndexMap weights, IndexMap biases) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < shape.count * shape.channels; i += gridDim.x * blockDim.x) {
        const auto c = i / shape.spatial % shape.channels;
        float value = (storage_cast<float>(input[layout.offset(i)]) - statistics[c]) * statistics[shape.channels + c];
        if (weight) value *= weight[weights.offset(c)];
        if (bias) value += bias[biases.offset(c)];
        output[i] = value;
    }
}

template <class T>
__global__ void batch_gradient_parts_kernel(const float* grad, const T* input, const float* statistics,
    float* partials, Channels shape, bool project, IndexMap gradients, IndexMap source) {
    __shared__ float shared[block_size];
    const auto tasks = shape.channels * shape.parts;
    for (Index task = blockIdx.x; task < tasks; task += gridDim.x) {
        const auto c = task % shape.channels, part = task / shape.channels;
        const auto begin = part * shape.count / shape.parts, end = (part + 1) * shape.count / shape.parts;
        float total = 0.0f, projection = 0.0f;
        for (Index i = begin + threadIdx.x; i < end; i += blockDim.x) {
            const auto position = shape.index(c, i);
            const float g = grad[gradients.offset(position)];
            total += g;
            if (project) projection += g * ((storage_cast<float>(input[source.offset(position)]) - statistics[c])
                * statistics[shape.channels + c]);
        }
        total = block_reduce<false>(total, shared);
        projection = block_reduce<false>(projection, shared);
        if (threadIdx.x == 0) {
            partials[task] = total;
            partials[tasks + task] = projection;
        }
    }
}

/// 两个通道归约同时供 dx、dw 和 db 使用，仿射梯度直接写入图中的累计目标。
__global__ void batch_gradient_sum_kernel(const float* partials, float* reductions,
    float* dw, float* db, Channels shape, bool add_weight, bool add_bias) {
    const auto tasks = shape.channels * shape.parts;
    for (Index c = blockIdx.x * blockDim.x + threadIdx.x; c < shape.channels; c += gridDim.x * blockDim.x) {
        float total = 0.0f, projection = 0.0f;
        for (Index part = 0; part < shape.parts; ++part) {
            total += partials[part * shape.channels + c];
            projection += partials[tasks + part * shape.channels + c];
        }
        reductions[c] = total / static_cast<float>(shape.count);
        reductions[shape.channels + c] = projection / static_cast<float>(shape.count);
        if (dw) dw[c] = (add_weight ? dw[c] : 0.0f) + projection;
        if (db) db[c] = (add_bias ? db[c] : 0.0f) + total;
    }
}

template <class T>
__global__ void batch_input_gradient_kernel(const float* grad, const T* input, const float* statistics,
    const float* reductions, const float* weight, float* output, Channels shape, bool accumulate,
    IndexMap gradients, IndexMap source, IndexMap weights) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < shape.count * shape.channels; i += gridDim.x * blockDim.x) {
        const auto c = i / shape.spatial % shape.channels;
        const float inverse = statistics[shape.channels + c];
        float value = grad[gradients.offset(i)];
        if (reductions) {
            const float normalized = (storage_cast<float>(input[source.offset(i)]) - statistics[c]) * inverse;
            value = value - reductions[c] - normalized * reductions[shape.channels + c];
        }
        value *= inverse * (weight ? weight[weights.offset(c)] : 1.0f);
        output[i] = (accumulate ? output[i] : 0.0f) + value;
    }
}

} // namespace

void batch_normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, std::optional<numeric::BatchNormalizationState>& state,
    bool training, double momentum, double epsilon, numeric::NormalizationResult& result) {
    Execution execution(input);
    const Channels shape(input.layout);
    std::optional<TensorData> partials;
    if (training || !state) {
        partials = empty_data({2, shape.parts, shape.channels}, DType::f32, input.device());
        visit_storage(input.dtype, [&]<class T>() {
            batch_moments_kernel<<<row_grid_size(shape.parts * shape.channels), block_size, 0, execution.stream()>>>(
                input.data<T>(), partials->mutable_data<float>(), shape, IndexMap(input.layout));
        });
    }
    batch_statistics_kernel<<<grid_size(shape.channels), block_size, 0, execution.stream()>>>(
        partials ? partials->data<float>() : nullptr, result.statistics->mutable_data<float>(),
        state ? state->mean.mutable_data<float>() : nullptr, state ? state->variance.mutable_data<float>() : nullptr,
        shape, static_cast<float>(epsilon), static_cast<float>(1.0 - momentum), static_cast<float>(momentum),
        static_cast<float>(momentum * shape.count / (shape.count - 1)),
        state ? IndexMap(state->mean.layout) : IndexMap{}, state ? IndexMap(state->variance.layout) : IndexMap{});
    if (input.layout.numel() > 0) visit_storage(input.dtype, [&]<class T>() {
        batch_affine_kernel<<<grid_size(input.layout.numel()), block_size, 0, execution.stream()>>>(
            input.data<T>(), result.statistics->data<float>(), weight ? weight->data<float>() : nullptr,
            bias ? bias->data<float>() : nullptr, result.output.mutable_data<float>(), shape, IndexMap(input.layout),
            weight ? IndexMap(weight->layout) : IndexMap{}, bias ? IndexMap(bias->layout) : IndexMap{});
    });
    check(cudaGetLastError());
}

void batch_normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    bool batch_statistics, numeric::NormalizationGradients& result) {
    Execution execution(input);
    const Channels shape(input.layout);
    std::optional<TensorData> reductions;
    if (result.weight || result.bias || (result.input && batch_statistics)) {
        auto partials = empty_data({2, shape.parts, shape.channels}, DType::f32, input.device());
        reductions = empty_data({2, shape.channels}, DType::f32, input.device());
        visit_storage(input.dtype, [&]<class T>() {
            batch_gradient_parts_kernel<<<row_grid_size(shape.parts * shape.channels), block_size, 0, execution.stream()>>>(
                grad.data<float>(), input.data<T>(), statistics.data<float>(), partials.mutable_data<float>(), shape,
                bool(result.weight) || (bool(result.input) && batch_statistics), IndexMap(grad.layout), IndexMap(input.layout));
        });
        batch_gradient_sum_kernel<<<grid_size(shape.channels), block_size, 0, execution.stream()>>>(
            partials.data<float>(), reductions->mutable_data<float>(),
            result.weight ? result.weight->mutable_data<float>() : nullptr, result.bias ? result.bias->mutable_data<float>() : nullptr,
            shape, result.accumulate[1], result.accumulate[2]);
    }
    if (result.input && input.layout.numel() > 0) visit_storage(input.dtype, [&]<class T>() {
        batch_input_gradient_kernel<<<grid_size(input.layout.numel()), block_size, 0, execution.stream()>>>(
            grad.data<float>(), input.data<T>(), statistics.data<float>(),
            batch_statistics ? reductions->data<float>() : nullptr, weight ? weight->data<float>() : nullptr,
            result.input->mutable_data<float>(), shape, result.accumulate[0], IndexMap(grad.layout), IndexMap(input.layout),
            weight ? IndexMap(weight->layout) : IndexMap{});
    });
    check(cudaGetLastError());
}

} // namespace dl::detail::cuda
