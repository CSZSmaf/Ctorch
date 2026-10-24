#include "api/common.h"

#include <algorithm>

using namespace dl::capi;

extern "C" {

dl_status dl_functional_create(const dl_module* module, dl_functional** output) {
    return make_handle(output, [&] { return module->value; });
}
void dl_functional_release(dl_functional* functional) { delete functional; }
size_t dl_functional_parameter_count(const dl_functional* functional) { return functional->value.parameter_names().size(); }
const char* dl_functional_parameter_name(const dl_functional* functional, size_t index) {
    return functional->value.parameter_names()[index].c_str();
}
size_t dl_functional_buffer_count(const dl_functional* functional) { return functional->value.buffer_names().size(); }
const char* dl_functional_buffer_name(const dl_functional* functional, size_t index) {
    return functional->value.buffer_names()[index].c_str();
}
size_t dl_functional_module_count(const dl_functional* functional) { return functional->value.module_names().size(); }
const char* dl_functional_module_name(const dl_functional* functional, size_t index) {
    return functional->value.module_names()[index].c_str();
}

dl_status dl_functional_state_create(const dl_functional* functional, dl_functional_state** output) {
    return make_handle(output, [&] { return functional->value.state(); });
}
void dl_functional_state_release(dl_functional_state* state) { delete state; }
dl_status dl_functional_state_parameters(const dl_functional_state* state, dl_tensor_list** output) {
    return make_handle(output, [&] { return state->value.parameters; });
}
dl_status dl_functional_state_buffers(const dl_functional_state* state, dl_tensor_list** output) {
    return make_handle(output, [&] { return state->value.buffers; });
}
dl_status dl_functional_state_set_parameter(dl_functional_state* state, size_t index, const dl_tensor* value) {
    return invoke([&] { state->value.parameters[index] = value->value; });
}
dl_status dl_functional_state_set_buffer(dl_functional_state* state, size_t index, const dl_tensor* value) {
    return invoke([&] { state->value.buffers[index] = value->value; });
}
int32_t dl_functional_state_training(const dl_functional_state* state, size_t module_index) {
    return state->value.training[module_index] != 0;
}
dl_status dl_functional_state_set_training(dl_functional_state* state, size_t module_index, int32_t enabled) {
    return invoke([&] { state->value.training[module_index] = enabled != 0; });
}
dl_status dl_functional_state_train(dl_functional_state* state, int32_t enabled) {
    return invoke([&] { std::fill(state->value.training.begin(), state->value.training.end(), enabled != 0); });
}

dl_status dl_functional_call(const dl_functional* functional, const dl_functional_state* state,
    const dl_tensor* input, dl_random_key key, dl_tensor** output, dl_functional_state** next_state) {
    *output = nullptr;
    *next_state = nullptr;
    return invoke([&] {
        auto result = functional->value.call(state->value, convert(key), [&] { return (*functional->owner)(input->value); });
        auto tensor = std::make_unique<dl_tensor>(dl_tensor{std::move(result.output)});
        auto next = std::make_unique<dl_functional_state>(dl_functional_state{
            {state->value.parameters, std::move(result.buffers), state->value.training}});
        *output = tensor.release();
        *next_state = next.release();
    });
}

} // extern "C"
