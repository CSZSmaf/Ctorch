#include "api/common.h"

#include "optim/result_tensors.h"

using namespace dl::capi;
using dl::optim::detail::result_tensors;

namespace {

std::optional<dl::optim::AdamMoments> moments(const dl_tensor* first, const dl_tensor* second, int64_t step) {
    return first ? std::optional{dl::optim::AdamMoments{first->value, second->value, step}} : std::nullopt;
}

std::optional<dl::optim::RMSpropBuffers> buffers(const dl_tensor* square_average,
    const dl_tensor* momentum, const dl_tensor* gradient_average, int64_t step) {
    return square_average ? std::optional{dl::optim::RMSpropBuffers{square_average->value,
        optional(momentum), optional(gradient_average), step}} : std::nullopt;
}

std::optional<dl::optim::AdagradAccumulator> accumulator(const dl_tensor* sum, int64_t step) {
    return sum ? std::optional{dl::optim::AdagradAccumulator{sum->value, step}} : std::nullopt;
}

std::vector<dl::optim::ParameterGroup> parameter_groups(const dl_parameter_group* groups, std::size_t count) {
    std::vector<dl::optim::ParameterGroup> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& group = groups[index];
        result.push_back({tensor_list(group.parameters, group.count),
            group.learning_rate ? std::optional{*group.learning_rate} : std::nullopt,
            group.weight_decay ? std::optional{*group.weight_decay} : std::nullopt});
    }
    return result;
}

} // namespace

extern "C" {

dl_adam_options dl_default_adam_options(void) {
    const dl::optim::AdamOptions v;
    return {v.learning_rate, v.beta1, v.beta2, v.epsilon, v.weight_decay};
}
dl_rmsprop_options dl_default_rmsprop_options(void) {
    const dl::optim::RMSpropOptions v;
    return {v.learning_rate, v.alpha, v.epsilon, v.weight_decay, v.momentum, v.centered};
}
dl_adagrad_options dl_default_adagrad_options(void) {
    const dl::optim::AdagradOptions v;
    return {v.learning_rate, v.learning_rate_decay, v.weight_decay, v.initial_accumulator, v.epsilon};
}

dl_status dl_sgd(const dl_tensor* parameter, const dl_tensor* gradient, const dl_tensor* velocity,
    dl_sgd_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        return result_tensors(dl::optim::sgd(parameter->value, gradient->value, optional(velocity), dl::capi::convert(options)));
    });
}

dl_status dl_adamw(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* first, const dl_tensor* second, int64_t step,
    dl_adamw_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        return result_tensors(dl::optim::adamw(parameter->value, gradient->value,
            moments(first, second, step), dl::capi::convert(options)));
    });
}

dl_status dl_sgd_update(dl_tensor* parameter, const dl_tensor* gradient, dl_tensor* velocity,
    dl_sgd_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto state = optional(velocity);
        dl::optim::sgd_(parameter->value, gradient->value, state, dl::capi::convert(options));
        return result_tensors(dl::optim::SGDResult{parameter->value, std::move(state)});
    });
}

dl_status dl_adamw_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* first, dl_tensor* second, int64_t step,
    dl_adamw_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto state = moments(first, second, step);
        dl::optim::adamw_(parameter->value, gradient->value, state, dl::capi::convert(options));
        return result_tensors(dl::optim::AdamWResult{parameter->value, std::move(*state)});
    });
}

dl_status dl_adam(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* first, const dl_tensor* second, int64_t step,
    dl_adam_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        return result_tensors(dl::optim::adam(parameter->value, gradient->value,
            moments(first, second, step), convert(options)));
    });
}

dl_status dl_adam_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* first, dl_tensor* second, int64_t step,
    dl_adam_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto state = moments(first, second, step);
        dl::optim::adam_(parameter->value, gradient->value, state, convert(options));
        return result_tensors(dl::optim::AdamResult{parameter->value, std::move(*state)});
    });
}

dl_status dl_rmsprop(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* square_average, const dl_tensor* momentum, const dl_tensor* gradient_average,
    int64_t step, dl_rmsprop_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        return result_tensors(dl::optim::rmsprop(parameter->value, gradient->value,
            buffers(square_average, momentum, gradient_average, step), convert(options)));
    });
}

dl_status dl_rmsprop_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* square_average, dl_tensor* momentum, dl_tensor* gradient_average,
    int64_t step, dl_rmsprop_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto state = buffers(square_average, momentum, gradient_average, step);
        dl::optim::rmsprop_(parameter->value, gradient->value, state, convert(options));
        return result_tensors(dl::optim::RMSpropResult{parameter->value, std::move(*state)});
    });
}

dl_status dl_adagrad(const dl_tensor* parameter, const dl_tensor* gradient,
    const dl_tensor* sum, int64_t step, dl_adagrad_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        return result_tensors(dl::optim::adagrad(parameter->value, gradient->value,
            accumulator(sum, step), convert(options)));
    });
}

dl_status dl_adagrad_update(dl_tensor* parameter, const dl_tensor* gradient,
    dl_tensor* sum, int64_t step, dl_adagrad_options options, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto state = accumulator(sum, step);
        dl::optim::adagrad_(parameter->value, gradient->value, state, convert(options));
        return result_tensors(dl::optim::AdagradResult{parameter->value, std::move(*state)});
    });
}

dl_status dl_sgd_create(const dl_tensor* const* parameters, size_t count, dl_sgd_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::SGD>(tensor_list(parameters, count), dl::capi::convert(options)); });
}
dl_status dl_adamw_create(const dl_tensor* const* parameters, size_t count, dl_adamw_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::AdamW>(tensor_list(parameters, count), dl::capi::convert(options)); });
}
dl_status dl_adam_create(const dl_tensor* const* parameters, size_t count, dl_adam_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::Adam>(tensor_list(parameters, count), convert(options)); });
}
dl_status dl_rmsprop_create(const dl_tensor* const* parameters, size_t count, dl_rmsprop_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::RMSprop>(tensor_list(parameters, count), convert(options)); });
}
dl_status dl_adagrad_create(const dl_tensor* const* parameters, size_t count, dl_adagrad_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::Adagrad>(tensor_list(parameters, count), convert(options)); });
}

dl_status dl_sgd_create_groups(const dl_parameter_group* groups, size_t count, dl_sgd_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::SGD>(parameter_groups(groups, count), dl::capi::convert(options)); });
}
dl_status dl_adamw_create_groups(const dl_parameter_group* groups, size_t count, dl_adamw_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::AdamW>(parameter_groups(groups, count), dl::capi::convert(options)); });
}
dl_status dl_adam_create_groups(const dl_parameter_group* groups, size_t count, dl_adam_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::Adam>(parameter_groups(groups, count), convert(options)); });
}
dl_status dl_rmsprop_create_groups(const dl_parameter_group* groups, size_t count, dl_rmsprop_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::RMSprop>(parameter_groups(groups, count), convert(options)); });
}
dl_status dl_adagrad_create_groups(const dl_parameter_group* groups, size_t count, dl_adagrad_options options, dl_optimizer** output) {
    return make_handle(output, [&] { return std::make_shared<dl::optim::Adagrad>(parameter_groups(groups, count), convert(options)); });
}

void dl_optimizer_release(dl_optimizer* optimizer) { delete optimizer; }
dl_status dl_optimizer_zero_grad(dl_optimizer* optimizer) { return invoke([&] { optimizer->value->zero_grad(); }); }
dl_status dl_optimizer_step(dl_optimizer* optimizer) { return invoke([&] { optimizer->value->step(); }); }
size_t dl_optimizer_group_count(const dl_optimizer* optimizer) { return optimizer->value->groups().size(); }
double dl_optimizer_learning_rate(const dl_optimizer* optimizer, size_t group) { return optimizer->value->learning_rate(group); }
double dl_optimizer_weight_decay(const dl_optimizer* optimizer, size_t group) { return optimizer->value->weight_decay(group); }
dl_status dl_optimizer_set_learning_rate(dl_optimizer* optimizer, size_t group, double value) {
    return invoke([&] { optimizer->value->set_learning_rate(value, group); });
}
dl_status dl_optimizer_set_weight_decay(dl_optimizer* optimizer, size_t group, double value) {
    return invoke([&] { optimizer->value->set_weight_decay(value, group); });
}
dl_status dl_clip_grad_norm(const dl_tensor* const* parameters, size_t count, double max_norm, double order, dl_tensor** output) {
    return make_handle(output, [&] { return dl::optim::clip_grad_norm(tensor_list(parameters, count), max_norm, order); });
}
dl_status dl_clip_grad_value(const dl_tensor* const* parameters, size_t count, double value) {
    return invoke([&] { dl::optim::clip_grad_value(tensor_list(parameters, count), value); });
}

dl_status dl_clip_gradients_norm(const dl_gradients* gradients, double max_norm, double order,
    dl_gradients** output, dl_tensor** total_norm) {
    *output = nullptr;
    *total_norm = nullptr;
    return invoke([&] {
        auto result = dl::optim::clip_gradients_norm(gradients->value, max_norm, order);
        auto values = std::make_unique<dl_gradients>(dl_gradients{std::move(result.gradients)});
        auto norm = std::make_unique<dl_tensor>(dl_tensor{std::move(result.total_norm)});
        *output = values.release();
        *total_norm = norm.release();
    });
}

dl_status dl_clip_gradients_norm_update(dl_gradients* gradients, double max_norm, double order,
    dl_tensor** total_norm) {
    return make_handle(total_norm, [&] {
        return dl::optim::clip_gradients_norm_(gradients->value, max_norm, order);
    });
}

dl_status dl_clip_gradients_value(const dl_gradients* gradients, double value, dl_gradients** output) {
    return make_handle(output, [&] { return dl::optim::clip_gradients_value(gradients->value, value); });
}

dl_status dl_clip_gradients_value_update(dl_gradients* gradients, double value) {
    return invoke([&] { dl::optim::clip_gradients_value_(gradients->value, value); });
}

} // extern "C"
