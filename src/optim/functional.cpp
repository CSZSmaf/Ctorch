#include "dl/optim.h"

#include "optim/update.h"

namespace dl::optim {
namespace {

namespace numeric = dl::detail::numeric;
using dl::detail::data_of;
using dl::detail::make_tensor;

Tensor copy_value(const Tensor& value, bool requires_grad = false) {
    return make_tensor(numeric::clone(data_of(value)), requires_grad);
}

std::optional<Tensor> copy_value(const std::optional<Tensor>& value) {
    return value ? std::optional{copy_value(*value)} : std::nullopt;
}

AdamMoments copy_value(const AdamMoments& value) {
    return {copy_value(value.first), copy_value(value.second), value.step};
}

RMSpropBuffers copy_value(const RMSpropBuffers& value) {
    return {copy_value(value.square_average), copy_value(value.momentum),
        copy_value(value.gradient_average), value.step};
}

AdagradAccumulator copy_value(const AdagradAccumulator& value) {
    return {copy_value(value.sum), value.step};
}

template <class State>
std::optional<State> copy_state(const std::optional<State>& value) {
    return value ? std::optional{copy_value(*value)} : std::nullopt;
}

} // namespace

void sgd_(Tensor parameter, const Tensor& gradient, std::optional<Tensor>& velocity, SGDOptions options) {
    const auto update = detail::prepare_sgd_update(parameter, gradient, velocity, options);
    numeric::sgd(std::span(&update, 1));
}

SGDResult sgd(const Tensor& parameter, const Tensor& gradient,
    const std::optional<Tensor>& velocity, SGDOptions options) {
    auto next = copy_value(parameter, parameter.requires_grad());
    auto state = velocity ? std::optional{copy_value(*velocity)} : std::nullopt;
    sgd_(next, gradient, state, options);
    return {std::move(next), std::move(state)};
}

namespace {

void adam_update(Tensor& parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, const AdamWOptions& options, bool decoupled) {
    const auto update = detail::prepare_adam_update(parameter, gradient, moments, options, decoupled);
    numeric::adam(std::span(&update, 1));
}

} // namespace

void adamw_(Tensor parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, AdamWOptions options) {
    adam_update(parameter, gradient, moments, options, true);
}

void adam_(Tensor parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, AdamOptions options) {
    // Adam 的衰减在逐元素更新中进入矩估计，与 AdamW 共用状态和数值入口。
    adam_update(parameter, gradient, moments,
        {options.learning_rate, options.beta1, options.beta2, options.epsilon, options.weight_decay}, false);
}

AdamWResult adamw(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdamMoments>& moments, AdamWOptions options) {
    auto next = copy_value(parameter, parameter.requires_grad());
    auto state = copy_state(moments);
    adamw_(next, gradient, state, options);
    return {std::move(next), std::move(*state)};
}

AdamResult adam(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdamMoments>& moments, AdamOptions options) {
    auto next = copy_value(parameter, parameter.requires_grad());
    auto state = copy_state(moments);
    adam_(next, gradient, state, options);
    return {std::move(next), std::move(*state)};
}

void rmsprop_(Tensor parameter, const Tensor& gradient,
    std::optional<RMSpropBuffers>& buffers, RMSpropOptions options) {
    const auto update = detail::prepare_rmsprop_update(parameter, gradient, buffers, options);
    numeric::rmsprop(std::span(&update, 1));
}

RMSpropResult rmsprop(const Tensor& parameter, const Tensor& gradient,
    const std::optional<RMSpropBuffers>& buffers, RMSpropOptions options) {
    auto next = copy_value(parameter, parameter.requires_grad());
    auto state = copy_state(buffers);
    rmsprop_(next, gradient, state, options);
    return {std::move(next), std::move(*state)};
}

void adagrad_(Tensor parameter, const Tensor& gradient,
    std::optional<AdagradAccumulator>& accumulator, AdagradOptions options) {
    const auto update = detail::prepare_adagrad_update(parameter, gradient, accumulator, options);
    numeric::adagrad(std::span(&update, 1));
}

AdagradResult adagrad(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdagradAccumulator>& accumulator, AdagradOptions options) {
    auto next = copy_value(parameter, parameter.requires_grad());
    auto state = copy_state(accumulator);
    adagrad_(next, gradient, state, options);
    return {std::move(next), std::move(*state)};
}

} // namespace dl::optim
