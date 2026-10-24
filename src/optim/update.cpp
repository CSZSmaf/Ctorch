#include "optim/update.h"

#include <cmath>

namespace dl::optim::detail {

namespace numeric = dl::detail::numeric;
using dl::detail::data_of;
using dl::detail::TensorAccess;

numeric::SGDUpdate prepare_sgd_update(Tensor& parameter, const Tensor& gradient,
    std::optional<Tensor>& velocity, const SGDOptions& options) {
    const bool initialize = options.momentum != 0.0 && !velocity;
    if (initialize) {
        // 首次更新在同一遍历中写满动量，分配时只建立存储，无需额外填零或复制方向。
        velocity = dl::detail::make_tensor(dl::detail::empty_data(parameter.shape(), parameter.dtype(), parameter.device()));
    }
    return {&TensorAccess::mutable_data(parameter), &data_of(gradient),
        options.momentum != 0.0 ? &TensorAccess::mutable_data(*velocity) : nullptr,
        {options.learning_rate, options.momentum, options.dampening, options.weight_decay,
         options.nesterov, initialize}};
}

numeric::AdamUpdate prepare_adam_update(Tensor& parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, const AdamWOptions& options, bool decoupled) {
    if (!moments) {
        const TensorOptions allocation{.dtype = parameter.dtype(), .device = parameter.device()};
        moments = AdamMoments{zeros(parameter.shape(), allocation), zeros(parameter.shape(), allocation), 0};
    }
    const auto step = ++moments->step;
    return {&TensorAccess::mutable_data(parameter), &data_of(gradient),
        &TensorAccess::mutable_data(moments->first), &TensorAccess::mutable_data(moments->second),
        {options.learning_rate, options.beta1, options.beta2, options.epsilon, options.weight_decay,
         1.0 / (1.0 - std::pow(options.beta1, step)), 1.0 / (1.0 - std::pow(options.beta2, step)), decoupled}};
}

numeric::RMSpropUpdate prepare_rmsprop_update(Tensor& parameter, const Tensor& gradient,
    std::optional<RMSpropBuffers>& buffers, const RMSpropOptions& options) {
    const TensorOptions allocation{.dtype = parameter.dtype(), .device = parameter.device()};
    auto zero = [&] { return zeros(parameter.shape(), allocation); };
    if (!buffers) buffers = RMSpropBuffers{zero(), {}, {}, 0};
    auto& state = *buffers;
    // 可选统计量按实际使用建立；关闭某一形式时保存此前的数值和身份。
    if (options.momentum != 0.0 && !state.momentum) state.momentum = zero();
    if (options.centered && !state.gradient_average) state.gradient_average = zero();
    ++state.step;
    return {&TensorAccess::mutable_data(parameter), &data_of(gradient),
        &TensorAccess::mutable_data(state.square_average),
        options.momentum != 0.0 ? &TensorAccess::mutable_data(*state.momentum) : nullptr,
        options.centered ? &TensorAccess::mutable_data(*state.gradient_average) : nullptr,
        {options.learning_rate, options.alpha, options.epsilon, options.weight_decay, options.momentum}};
}

numeric::AdagradUpdate prepare_adagrad_update(Tensor& parameter, const Tensor& gradient,
    std::optional<AdagradAccumulator>& accumulator, const AdagradOptions& options) {
    if (!accumulator) {
        accumulator = AdagradAccumulator{full(parameter.shape(), options.initial_accumulator,
            {.dtype = parameter.dtype(), .device = parameter.device()}), 0};
    }
    auto& state = *accumulator;
    ++state.step;
    const double rate = options.learning_rate / (1.0 + (state.step - 1) * options.learning_rate_decay);
    return {&TensorAccess::mutable_data(parameter), &data_of(gradient),
        &TensorAccess::mutable_data(state.sum), {rate, options.epsilon, options.weight_decay}};
}

void step_adam_parameters(std::span<Tensor> parameters, std::span<const ParameterGroupState> groups,
    std::vector<std::optional<AdamMoments>>& states, const AdamWOptions& defaults, bool decoupled) {
    batch_parameters(parameters, groups, states, defaults,
        [decoupled](auto& parameter, const auto& gradient, auto& state, const auto& options) {
            return prepare_adam_update(parameter, gradient, state, options, decoupled);
        }, numeric::adam);
}

} // namespace dl::optim::detail
