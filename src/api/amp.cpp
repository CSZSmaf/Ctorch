#include "api/common.h"

using namespace dl::capi;

namespace {

dl::GradScalerOptions options_from_c(dl_grad_scaler_options value) {
    return {value.initial_scale, value.growth_factor, value.backoff_factor, value.growth_interval, value.enabled != 0};
}

dl_grad_scaler_options options_to_c(const dl::GradScalerOptions& value) {
    return {value.initial_scale, value.growth_factor, value.backoff_factor, value.growth_interval, value.enabled};
}

} // namespace

extern "C" {

dl_grad_scaler_options dl_default_grad_scaler_options(void) { return options_to_c(dl::GradScalerOptions{}); }

dl_status dl_grad_scaler_create(dl_grad_scaler_options options, dl_grad_scaler** output) {
    return make_handle(output, [&] { return dl::GradScaler(options_from_c(options)); });
}

void dl_grad_scaler_release(dl_grad_scaler* scaler) { delete scaler; }
double dl_grad_scaler_get_scale(const dl_grad_scaler* scaler) { return scaler->value.get_scale(); }

dl_status dl_grad_scaler_scale(const dl_grad_scaler* scaler, const dl_tensor* loss, dl_tensor** output) {
    return make_handle(output, [&] { return scaler->value.scale(loss->value); });
}

dl_status dl_grad_scaler_unscale(const dl_grad_scaler* scaler, dl_optimizer* optimizer, int32_t* finite) {
    return invoke([&] { *finite = scaler->value.unscale(*optimizer->value); });
}

dl_status dl_grad_scaler_unscale_gradients(const dl_grad_scaler* scaler,
    const dl_gradients* gradients, dl_gradients** output, int32_t* finite) {
    return make_handle(output, [&] {
        auto result = scaler->value.unscale(gradients->value);
        *finite = result.finite;
        return std::move(result.gradients);
    });
}

dl_status dl_grad_scaler_unscale_gradients_update(const dl_grad_scaler* scaler,
    dl_gradients* gradients, int32_t* finite) {
    return invoke([&] { *finite = scaler->value.unscale_(gradients->value); });
}

dl_status dl_grad_scaler_update(dl_grad_scaler* scaler, int32_t finite) {
    return invoke([&] { scaler->value.update(finite != 0); });
}

dl_status dl_grad_scaler_step(dl_grad_scaler* scaler, dl_optimizer* optimizer, int32_t* updated) {
    return invoke([&] { *updated = scaler->value.step(*optimizer->value); });
}

dl_status dl_grad_scaler_get_state(const dl_grad_scaler* scaler, dl_grad_scaler_state* output) {
    return invoke([&] {
        const auto state = scaler->value.state_dict();
        *output = {options_to_c(state.options), state.scale, state.finite_steps};
    });
}

dl_status dl_grad_scaler_load_state(dl_grad_scaler* scaler, const dl_grad_scaler_state* state) {
    return invoke([&] {
        scaler->value.load_state_dict({options_from_c(state->options), state->scale, state->finite_steps});
    });
}

dl_status dl_checkpoint_set_grad_scaler(dl_checkpoint* checkpoint, const dl_grad_scaler* scaler) {
    return invoke([&] { checkpoint->value.scaler = scaler->value.state_dict(); });
}

dl_status dl_checkpoint_restore_grad_scaler(const dl_checkpoint* checkpoint, dl_grad_scaler* scaler, int32_t* restored) {
    return invoke([&] {
        *restored = checkpoint->value.scaler.has_value();
        if (*restored) scaler->value.load_state_dict(*checkpoint->value.scaler);
    });
}

} // extern "C"
