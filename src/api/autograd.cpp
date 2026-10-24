#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_status dl_grad(const dl_tensor* output, const dl_tensor* const* inputs,
    size_t input_count, const dl_tensor* seed, dl_gradients** gradients) {
    return make_handle(gradients, [&] {
        return dl::grad(output->value, tensor_list(inputs, input_count), optional(seed));
    });
}

dl_status dl_grad_many(const dl_tensor* const* outputs, size_t output_count,
    const dl_tensor* const* inputs, size_t input_count, const dl_tensor* const* seeds,
    dl_gradients** gradients) {
    return make_handle(gradients, [&] {
        auto results = tensor_list(outputs, output_count);
        auto arguments = tensor_list(inputs, input_count);
        auto cotangents = seeds ? tensor_list(seeds, output_count) : std::vector<dl::Tensor>{};
        return dl::grad(results, arguments, cotangents);
    });
}

dl_status dl_gradients_create(const dl_tensor* const* values, size_t count, dl_gradients** output) {
    return make_handle(output, [&] {
        dl::Gradients result(count);
        for (std::size_t i = 0; i < count; ++i) result[i] = optional(values[i]);
        return result;
    });
}

void dl_gradients_release(dl_gradients* gradients) { delete gradients; }
size_t dl_gradients_size(const dl_gradients* gradients) { return gradients->value.size(); }
dl_status dl_gradients_get(const dl_gradients* gradients, size_t index, dl_tensor** output) {
    *output = nullptr;
    return invoke([&] {
        if (const auto& gradient = gradients->value[index]) *output = new dl_tensor{*gradient};
    });
}

} // extern "C"
