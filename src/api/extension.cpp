#include "api/callback.h"

#include <stdexcept>

/// 回调中的规格与保存值可读；writer 仅在前向回调的生命周期内提供。
struct dl_vjp_context {
    const dl::VjpContext* value;
    dl::VjpContext* writer = nullptr;
};

extern "C" {

dl_tensor_spec dl_tensor_get_spec(const dl_tensor* tensor) {
    const auto& value = tensor->value;
    return {{value.shape().data(), value.shape().size()}, {value.strides().data(), value.strides().size()},
        static_cast<dl_dtype>(value.dtype()), dl::capi::convert(value.device()), value.storage_offset(),
        value.is_read_only()};
}

dl_status dl_custom_vjp(const dl_tensor* const* inputs, size_t count,
    dl_vjp_forward_fn forward, dl_vjp_backward_fn backward, void* user_data,
    dl_user_data_deleter deleter, dl_tensor** output) {
    *output = nullptr;
    return dl::capi::invoke([&] {
        auto state = dl::capi::callback_state(user_data, deleter);
        auto arguments = dl::capi::tensor_list(inputs, count);
        auto value = dl::custom_vjp(arguments,
            [forward, inputs, count, state](dl::VjpContext& context, std::span<const dl::Tensor>) {
                dl_vjp_context handle{&context, &context};
                dl::capi::ReturnedHandle<dl_tensor> result;
                const char* error = nullptr;
                if (forward(&handle, inputs, count, state.get(), &result.value, &error) != DL_OK)
                    throw std::runtime_error(error ? error : "custom VJP forward failed");
                return std::move(result.value->value);
            },
            [backward, count, state](const dl::VjpContext& context, const dl::Tensor& grad) {
                const dl_vjp_context handle{&context};
                const dl_tensor upstream{grad};
                dl::capi::ReturnedTensors result(count);
                const char* error = nullptr;
                if (backward(&handle, &upstream, state.get(), result.values.data(), count, &error) != DL_OK)
                    throw std::runtime_error(error ? error : "custom VJP backward failed");
                dl::VjpGradients gradients(count);
                for (std::size_t index = 0; index < count; ++index)
                    if (result.values[index]) gradients[index] = std::move(result.values[index]->value);
                return gradients;
            });
        *output = new dl_tensor{std::move(value)};
    });
}

size_t dl_vjp_input_count(const dl_vjp_context* context) { return context->value->input_count(); }
int32_t dl_vjp_needs_input_grad(const dl_vjp_context* context, size_t index) {
    return context->value->needs_input_grad(index);
}
dl_tensor_spec dl_vjp_input_spec(const dl_vjp_context* context, size_t index) {
    return dl::capi::convert(context->value->input_spec(index));
}

dl_status dl_vjp_save_for_backward(dl_vjp_context* context, const dl_tensor* const* tensors, size_t count) {
    return dl::capi::invoke([&] {
        auto values = dl::capi::tensor_list(tensors, count);
        context->writer->save_for_backward(values);
    });
}
size_t dl_vjp_saved_count(const dl_vjp_context* context) { return context->value->saved_tensors().size(); }
dl_status dl_vjp_saved_tensor(const dl_vjp_context* context, size_t index, dl_tensor** output) {
    return dl::capi::make_handle(output, [&] { return context->value->saved_tensors()[index]; });
}

dl_checkpoint_options dl_default_checkpoint_options(void) { return {1}; }

dl_status dl_activation_checkpoint(const dl_tensor* const* inputs, size_t count,
    dl_recompute_fn function, void* user_data, dl_user_data_deleter deleter,
    dl_checkpoint_options options, dl_tensor** output) {
    *output = nullptr;
    return dl::capi::invoke([&] {
        auto state = dl::capi::callback_state(user_data, deleter);
        auto arguments = dl::capi::tensor_list(inputs, count);
        auto value = dl::checkpoint(arguments,
            [function, state](std::span<const dl::Tensor> values) {
                // 每次从本次 Tensor 构造借用句柄，重算时由此传递新的叶身份。
                dl::capi::BorrowedTensors borrowed(values);
                dl::capi::ReturnedHandle<dl_tensor> result;
                const char* error = nullptr;
                if (function(borrowed.data(), borrowed.size(), state.get(), &result.value, &error) != DL_OK)
                    throw std::runtime_error(error ? error : "activation checkpoint callback failed");
                return std::move(result.value->value);
            }, {.preserve_rng_state = options.preserve_rng_state != 0});
        *output = new dl_tensor{std::move(value)};
    });
}

} // extern "C"
