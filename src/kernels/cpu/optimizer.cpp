#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"
#include "kernels/cpu/vector.h"

#include <cmath>

namespace dl::detail::cpu {

void momentum(TensorData& buffer, const TensorData& gradient, double decay, double scale) {
    IterationPlan<2> plan(buffer.layout.sizes, {buffer.layout.strides, gradient.layout.strides});
    visit_floating(buffer.dtype, [&]<class T> {
        auto* values = buffer.mutable_data<T>();
        const auto* update = gradient.data<T>();
        plan.for_each([&](Index, const auto& offset) {
            values[offset[0]] = static_cast<T>(decay) * values[offset[0]]
                + static_cast<T>(scale) * update[offset[1]];
        });
    });
}

void sgd(std::span<const numeric::SGDUpdate> updates) {
    for (const auto& update : updates) {
        auto& parameter = *update.parameter;
        const auto& gradient = *update.gradient;
        const auto& step = update.step;
        if (!update.velocity && step.weight_decay == 0.0) {
            axpy(parameter, gradient, -step.learning_rate);
            continue;
        }
        IterationPlan<3> plan(parameter.layout.sizes, {parameter.layout.strides, gradient.layout.strides,
            update.velocity ? update.velocity->layout.strides : parameter.layout.strides});
        visit_floating(parameter.dtype, [&]<class T> {
            auto* values = parameter.mutable_data<T>();
            const auto* gradients = gradient.data<T>();
            auto* velocities = update.velocity ? update.velocity->mutable_data<T>() : nullptr;
            const T rate = static_cast<T>(step.learning_rate), decay = static_cast<T>(step.weight_decay);
            const T momentum = static_cast<T>(step.momentum), weight = static_cast<T>(1.0 - step.dampening);
            plan.for_each([&](Index, const auto& offset) {
                T& value = values[offset[0]];
                T direction = gradients[offset[1]];
                if (step.weight_decay != 0.0) direction += decay * value;
                if (velocities) {
                    T& velocity = velocities[offset[2]];
                    // 首次动量直接取方向，已有缓冲从本步开始使用 dampening 系数。
                    velocity = step.initialize_velocity ? direction : momentum * velocity + weight * direction;
                    direction = step.nesterov ? direction + momentum * velocity : velocity;
                }
                value -= rate * direction;
            });
        });
    }
}

namespace {

void adam_one(const numeric::AdamUpdate& update) {
    auto& parameter = *update.parameter;
    const auto& gradient = *update.gradient;
    auto& first = *update.first;
    auto& second = *update.second;
    const auto& step = update.step;
    if (step.decoupled && parameter.layout.is_contiguous() && gradient.layout.is_contiguous()
        && first.layout.is_contiguous() && second.layout.is_contiguous()) {
        const auto count = static_cast<std::size_t>(parameter.layout.numel());
        if (parameter.dtype == DType::f32) {
            dl_adamw_f32(count, parameter.mutable_data<float>(), gradient.data<float>(),
                         first.mutable_data<float>(), second.mutable_data<float>(),
                         static_cast<float>(step.learning_rate), step.beta1,
                         step.beta2, static_cast<float>(step.epsilon),
                         static_cast<float>(step.weight_decay), static_cast<float>(step.inverse_bias1),
                         static_cast<float>(step.inverse_bias2));
        } else {
            dl_adamw_f64(count, parameter.mutable_data<double>(), gradient.data<double>(),
                         first.mutable_data<double>(), second.mutable_data<double>(),
                         step.learning_rate, step.beta1, step.beta2, step.epsilon,
                         step.weight_decay, step.inverse_bias1, step.inverse_bias2);
        }
        return;
    }
    IterationPlan<4> plan(parameter.layout.sizes, {parameter.layout.strides, gradient.layout.strides,
                                                  first.layout.strides, second.layout.strides});
    visit_floating(parameter.dtype, [&]<class T> {
        auto* values = parameter.mutable_data<T>();
        const auto* updates = gradient.data<T>();
        auto* moments = first.mutable_data<T>();
        auto* squares = second.mutable_data<T>();
        const T rate = static_cast<T>(step.learning_rate);
        const T beta1 = static_cast<T>(step.beta1);
        const T beta2 = static_cast<T>(step.beta2);
        const T weight1 = static_cast<T>(1.0 - step.beta1), weight2 = static_cast<T>(1.0 - step.beta2);
        const T decay = step.decoupled ? T{1} - rate * static_cast<T>(step.weight_decay) : T{1};
        plan.for_each([&](Index, const auto& offset) {
            T g = updates[offset[1]];
            if (!step.decoupled && step.weight_decay != 0.0)
                g += static_cast<T>(step.weight_decay) * values[offset[0]];
            T& m = moments[offset[2]];
            T& v = squares[offset[3]];
            m = beta1 * m + weight1 * g;
            v = beta2 * v + weight2 * g * g;
            const T update = m * static_cast<T>(step.inverse_bias1)
                / (std::sqrt(v * static_cast<T>(step.inverse_bias2)) + static_cast<T>(step.epsilon));
            values[offset[0]] = decay * values[offset[0]] - rate * update;
        });
    });
}

void rmsprop_one(const numeric::RMSpropUpdate& update) {
    auto& parameter = *update.parameter;
    const auto& gradient = *update.gradient;
    auto& square_average = *update.square_average;
    auto* momentum = update.momentum;
    auto* gradient_average = update.gradient_average;
    const auto& step = update.step;
    IterationPlan<5> plan(parameter.layout.sizes, {parameter.layout.strides, gradient.layout.strides,
        square_average.layout.strides, momentum ? momentum->layout.strides : parameter.layout.strides,
        gradient_average ? gradient_average->layout.strides : parameter.layout.strides});
    visit_floating(parameter.dtype, [&]<class T> {
        auto* values = parameter.mutable_data<T>();
        const auto* gradients = gradient.data<T>();
        auto* squares = square_average.mutable_data<T>();
        auto* velocities = momentum ? momentum->mutable_data<T>() : nullptr;
        auto* averages = gradient_average ? gradient_average->mutable_data<T>() : nullptr;
        const T rate = static_cast<T>(step.learning_rate), decay = static_cast<T>(step.weight_decay);
        const T alpha = static_cast<T>(step.alpha), weight = static_cast<T>(1.0 - step.alpha);
        const T epsilon = static_cast<T>(step.epsilon), mu = static_cast<T>(step.momentum);
        plan.for_each([&](Index, const auto& offset) {
            T& value = values[offset[0]];
            T g = gradients[offset[1]];
            if (step.weight_decay != 0.0) g += decay * value;
            T& square = squares[offset[2]];
            square = alpha * square + weight * g * g;
            T variance = square;
            if (averages) {
                T& average = averages[offset[4]];
                average = alpha * average + weight * g;
                variance -= average * average;
            }
            // centered 使用本步梯度均值；epsilon 位于平方根外。
            T direction = g / (std::sqrt(variance) + epsilon);
            if (velocities) {
                T& velocity = velocities[offset[3]];
                velocity = mu * velocity + direction;
                direction = velocity;
            }
            value -= rate * direction;
        });
    });
}

void adagrad_one(const numeric::AdagradUpdate& update) {
    auto& parameter = *update.parameter;
    const auto& gradient = *update.gradient;
    auto& sum = *update.sum;
    const auto& step = update.step;
    IterationPlan<3> plan(parameter.layout.sizes, {parameter.layout.strides, gradient.layout.strides, sum.layout.strides});
    visit_floating(parameter.dtype, [&]<class T> {
        auto* values = parameter.mutable_data<T>();
        const auto* gradients = gradient.data<T>();
        auto* sums = sum.mutable_data<T>();
        const T rate = static_cast<T>(step.learning_rate), decay = static_cast<T>(step.weight_decay);
        const T epsilon = static_cast<T>(step.epsilon);
        plan.for_each([&](Index, const auto& offset) {
            T& value = values[offset[0]];
            T g = gradients[offset[1]];
            if (step.weight_decay != 0.0) g += decay * value;
            T& accumulated = sums[offset[2]];
            accumulated += g * g;
            value -= rate * (g / (std::sqrt(accumulated) + epsilon));
        });
    });
}

} // namespace

void adam(std::span<const numeric::AdamUpdate> updates) {
    for (const auto& update : updates) adam_one(update);
}

void rmsprop(std::span<const numeric::RMSpropUpdate> updates) {
    for (const auto& update : updates) rmsprop_one(update);
}

void adagrad(std::span<const numeric::AdagradUpdate> updates) {
    for (const auto& update : updates) adagrad_one(update);
}

} // namespace dl::detail::cpu
