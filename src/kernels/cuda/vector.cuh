#pragma once

#include "kernels/cuda/common.cuh"
#include "ops/numeric.h"

namespace dl::detail::cuda {
namespace {

/// 向量更新只接收数据地址、逻辑布局与数值系数，输出沿给定布局写回。
__global__ void axpy_kernel(float* target, const float* input, Index count, IndexMap output,
                           IndexMap source, float alpha) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        target[output.offset(index)] += alpha * input[source.offset(index)];
    }
}

__global__ void momentum_kernel(float* buffer, const float* gradient, Index count,
                               IndexMap target, IndexMap source, float decay, float scale) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto position = target.offset(index);
        buffer[position] = decay * buffer[position] + scale * gradient[source.offset(index)];
    }
}

struct SGDArithmetic {
    float learning_rate, momentum, weight, weight_decay;
    bool use_decay, nesterov, initialize_velocity;

    __device__ explicit SGDArithmetic(const numeric::SGDStep& step)
        : learning_rate(static_cast<float>(step.learning_rate)), momentum(static_cast<float>(step.momentum)),
          weight(static_cast<float>(1.0 - step.dampening)), weight_decay(static_cast<float>(step.weight_decay)),
          use_decay(step.weight_decay != 0.0), nesterov(step.nesterov), initialize_velocity(step.initialize_velocity) {}

    __device__ void operator()(float& parameter, float direction, float* velocity) const {
        if (use_decay) direction += weight_decay * parameter;
        if (velocity) {
            *velocity = initialize_velocity ? direction : momentum * *velocity + weight * direction;
            direction = nesterov ? direction + momentum * *velocity : *velocity;
        }
        parameter -= learning_rate * direction;
    }
};

__global__ void sgd_kernel(float* parameter, const float* gradient, float* velocity, Index count,
                          IndexMap values, IndexMap updates, IndexMap velocities, numeric::SGDStep step) {
    const SGDArithmetic update(step);
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        update(parameter[values.offset(index)], gradient[updates.offset(index)],
            velocity ? velocity + velocities.offset(index) : nullptr);
    }
}

/// 单参数布局内核和连续参数批量内核共用每个元素的矩估计与衰减公式。
struct AdamArithmetic {
    float learning_rate, beta1, beta2, weight1, weight2, epsilon, weight_decay, inverse_bias1, inverse_bias2, decay;
    bool coupled_decay;

    __device__ explicit AdamArithmetic(const numeric::AdamStep& step)
        : learning_rate(static_cast<float>(step.learning_rate)), beta1(static_cast<float>(step.beta1)),
          beta2(static_cast<float>(step.beta2)), weight1(static_cast<float>(1.0 - step.beta1)),
          weight2(static_cast<float>(1.0 - step.beta2)), epsilon(static_cast<float>(step.epsilon)),
          weight_decay(static_cast<float>(step.weight_decay)), inverse_bias1(static_cast<float>(step.inverse_bias1)),
          inverse_bias2(static_cast<float>(step.inverse_bias2)),
          decay(step.decoupled ? 1.0f - learning_rate * weight_decay : 1.0f),
          coupled_decay(!step.decoupled && step.weight_decay != 0.0) {}

    __device__ void operator()(float& parameter, float gradient, float& first, float& second) const {
        if (coupled_decay) gradient += weight_decay * parameter;
        // 差值先按超参数精度计算，再转换到累积类型，与偏差校正使用同一组系数。
        first = beta1 * first + weight1 * gradient;
        second = beta2 * second + weight2 * gradient * gradient;
        const float update = first * inverse_bias1 / (sqrtf(second * inverse_bias2) + epsilon);
        parameter = decay * parameter - learning_rate * update;
    }
};

__global__ void adam_kernel(float* parameter, const float* gradient, float* first, float* second,
                           Index count, IndexMap values, IndexMap updates, IndexMap moments,
                           IndexMap squares, numeric::AdamStep step) {
    const AdamArithmetic update(step);
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        update(parameter[values.offset(index)], gradient[updates.offset(index)],
            first[moments.offset(index)], second[squares.offset(index)]);
    }
}

struct RMSpropArithmetic {
    float learning_rate, alpha, weight, epsilon, weight_decay, momentum;
    bool use_decay;

    __device__ explicit RMSpropArithmetic(const numeric::RMSpropStep& step)
        : learning_rate(static_cast<float>(step.learning_rate)), alpha(static_cast<float>(step.alpha)),
          weight(static_cast<float>(1.0 - step.alpha)), epsilon(static_cast<float>(step.epsilon)),
          weight_decay(static_cast<float>(step.weight_decay)), momentum(static_cast<float>(step.momentum)),
          use_decay(step.weight_decay != 0.0) {}

    __device__ void operator()(float& parameter, float g, float& square,
                               float* velocity, float* gradient_average) const {
        if (use_decay) g += weight_decay * parameter;
        square = alpha * square + weight * g * g;
        float variance = square;
        if (gradient_average) {
            float& average = *gradient_average;
            average = alpha * average + weight * g;
            variance -= average * average;
        }
        float direction = g / (sqrtf(variance) + epsilon);
        if (velocity) {
            *velocity = momentum * *velocity + direction;
            direction = *velocity;
        }
        parameter -= learning_rate * direction;
    }
};

__global__ void rmsprop_kernel(float* parameter, const float* gradient, float* square_average,
                              float* momentum, float* gradient_average, Index count,
                              IndexMap values, IndexMap gradients, IndexMap squares,
                              IndexMap velocities, IndexMap averages, numeric::RMSpropStep step) {
    const RMSpropArithmetic update(step);
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        update(parameter[values.offset(index)], gradient[gradients.offset(index)], square_average[squares.offset(index)],
            momentum ? momentum + velocities.offset(index) : nullptr,
            gradient_average ? gradient_average + averages.offset(index) : nullptr);
    }
}

struct AdagradArithmetic {
    float learning_rate, epsilon, weight_decay;
    bool use_decay;

    __device__ explicit AdagradArithmetic(const numeric::AdagradStep& step)
        : learning_rate(static_cast<float>(step.learning_rate)), epsilon(static_cast<float>(step.epsilon)),
          weight_decay(static_cast<float>(step.weight_decay)), use_decay(step.weight_decay != 0.0) {}

    __device__ void operator()(float& parameter, float gradient, float& sum) const {
        if (use_decay) gradient += weight_decay * parameter;
        sum += gradient * gradient;
        parameter -= learning_rate * (gradient / (sqrtf(sum) + epsilon));
    }
};

__global__ void adagrad_kernel(float* parameter, const float* gradient, float* sum, Index count,
                              IndexMap values, IndexMap gradients, IndexMap sums, numeric::AdagradStep step) {
    const AdagradArithmetic update(step);
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        update(parameter[values.offset(index)], gradient[gradients.offset(index)], sum[sums.offset(index)]);
    }
}

} // namespace
} // namespace dl::detail::cuda
