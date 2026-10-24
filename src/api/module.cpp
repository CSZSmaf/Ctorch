#include "api/common.h"

using namespace dl::capi;

extern "C" {

void dl_module_release(dl_module* module) { delete module; }

dl_status dl_module_retain(const dl_module* module, dl_module** output) {
    return make_handle(output, [&] { return module->value; });
}

dl_status dl_sequential_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Sequential>(); });
}

dl_status dl_sequential_append(dl_module* sequential, const char* name, const dl_module* child) {
    return invoke([&] {
        static_cast<dl::nn::Sequential&>(*sequential->value).append(name, child->value);
    });
}

dl_status dl_linear_create(int64_t input_features, int64_t output_features, int32_t bias,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Linear>(input_features, output_features,
            bias != 0, convert(options));
    });
}

dl_status dl_linear_from_tensors(const dl_tensor* weight, const dl_tensor* bias, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Linear>(weight->value, optional(bias));
    });
}

dl_status dl_embedding_create(int64_t vocabulary, int64_t dimensions, dl_tensor_options options,
    dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Embedding>(vocabulary, dimensions, convert(options));
    });
}

dl_status dl_embedding_from_tensor(const dl_tensor* weight, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Embedding>(weight->value); });
}

dl_status dl_relu_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::ReLU>(); });
}

dl_status dl_gelu_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::GELU>(); });
}

dl_status dl_sigmoid_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Sigmoid>(); });
}
dl_status dl_tanh_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Tanh>(); });
}
dl_status dl_silu_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::SiLU>(); });
}
dl_status dl_softplus_create(dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Softplus>(); });
}
dl_status dl_leaky_relu_create(double negative_slope, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::LeakyReLU>(negative_slope); });
}
dl_status dl_softmax_create(int64_t axis, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Softmax>(axis); });
}
dl_status dl_log_softmax_create(int64_t axis, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::LogSoftmax>(axis); });
}

dl_status dl_flatten_create(int64_t first, int64_t last, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Flatten>(first, last); });
}

dl_status dl_dropout_create(double probability, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::Dropout>(probability); });
}

dl_status dl_layer_norm_create(dl_shape normalized_shape, double epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::LayerNorm>(convert(normalized_shape), epsilon,
            affine != 0, convert(options));
    });
}

dl_status dl_conv2d_create(int64_t input_channels, int64_t output_channels, dl_size2 kernel,
    dl_conv2d_options convolution, int32_t bias, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Conv2d>(input_channels, output_channels, convert(kernel),
            convert(convolution), bias != 0, convert(options));
    });
}

dl_status dl_conv2d_from_tensors(const dl_tensor* weight, const dl_tensor* bias,
    dl_conv2d_options convolution, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Conv2d>(weight->value, optional(bias), convert(convolution));
    });
}

dl_status dl_max_pool2d_create(dl_pool2d_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::MaxPool2d>(convert(options)); });
}

dl_status dl_avg_pool2d_create(dl_pool2d_options options, int32_t count_include_pad, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::AvgPool2d>(convert(options), count_include_pad != 0);
    });
}

dl_status dl_batch_norm2d_create(int64_t channels, dl_batch_norm_options normalization,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::BatchNorm2d>(channels, convert(normalization), convert(options));
    });
}

dl_status dl_batch_norm2d_reset(dl_module* module) {
    return invoke([&] { static_cast<dl::nn::BatchNorm2d&>(*module->value).reset_running_stats(); });
}

dl_status dl_attention_create(int64_t dimensions, int64_t heads, dl_attention_options attention,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::MultiHeadAttention>(dimensions, heads, convert(attention), convert(options));
    });
}

dl_status dl_transformer_layer_create(int64_t dimensions, int64_t heads, int64_t feedforward,
    double dropout, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::TransformerEncoderLayer>(dimensions, heads, feedforward,
            dropout, convert(options));
    });
}

dl_status dl_attention_set_backend(dl_module* module, dl_attention_backend backend, int32_t causal) {
    return invoke([&] { static_cast<dl::nn::MultiHeadAttention&>(*module->value)
        .set_attention_backend(static_cast<dl::AttentionBackend>(backend), causal != 0); });
}
dl_status dl_transformer_layer_set_attention_backend(dl_module* module,
    dl_attention_backend backend, int32_t causal) {
    return invoke([&] { static_cast<dl::nn::TransformerEncoderLayer&>(*module->value)
        .set_attention_backend(static_cast<dl::AttentionBackend>(backend), causal != 0); });
}
dl_status dl_transformer_set_attention_backend(dl_module* module, dl_attention_backend backend, int32_t causal) {
    return invoke([&] { static_cast<dl::nn::TransformerEncoder&>(*module->value)
        .set_attention_backend(static_cast<dl::AttentionBackend>(backend), causal != 0); });
}

dl_status dl_transformer_create(int64_t layers, int64_t dimensions, int64_t heads, int64_t feedforward,
    double dropout, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::TransformerEncoder>(layers, dimensions, heads, feedforward,
            dropout, convert(options));
    });
}

dl_status dl_module_forward(dl_module* module, const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return module->value->forward(input->value); });
}

dl_status dl_attention_forward(dl_module* module, const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_tensor** output) {
    return make_handle(output, [&] {
        return static_cast<dl::nn::MultiHeadAttention&>(*module->value).forward(
            query->value, key->value, value->value, optional(keep_mask));
    });
}

dl_status dl_transformer_layer_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* keep_mask, dl_tensor** output) {
    return make_handle(output, [&] {
        return static_cast<dl::nn::TransformerEncoderLayer&>(*module->value).forward(
            input->value, optional(keep_mask));
    });
}

dl_status dl_transformer_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* keep_mask, dl_tensor** output) {
    return make_handle(output, [&] {
        return static_cast<dl::nn::TransformerEncoder&>(*module->value).forward(input->value, optional(keep_mask));
    });
}

dl_status dl_module_train(dl_module* module, int32_t training) {
    return invoke([&] { module->value->train(training != 0); });
}
int32_t dl_module_is_training(const dl_module* module) { return module->value->is_training(); }
dl_status dl_module_set_requires_grad(dl_module* module, int32_t enabled) {
    return invoke([&] { module->value->requires_grad_(enabled != 0); });
}
dl_status dl_module_to(dl_module* module, dl_device device) {
    return invoke([&] { module->value->to(convert(device)); });
}

dl_status dl_module_parameters(const dl_module* module, dl_state** output) {
    return make_handle(output, [&] { return module->value->named_parameters(); });
}
dl_status dl_module_trainable_parameters(const dl_module* module, dl_state** output) {
    return make_handle(output, [&] { return module->value->named_trainable_parameters(); });
}

dl_status dl_module_state(const dl_module* module, dl_state** output) {
    return make_handle(output, [&] { return module->value->state_dict(); });
}

dl_status dl_module_load_state(dl_module* module, const dl_state* state) {
    return invoke([&] { module->value->load_state_dict(state->value); });
}

} // extern "C"
