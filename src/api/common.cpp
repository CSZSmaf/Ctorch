#include "api/common.h"

#include <cstdio>
#include <cstring>

namespace dl::capi {
namespace {
thread_local char last_error[2048]{};
}

void set_error(const char* message) noexcept { std::snprintf(last_error, sizeof(last_error), "%s", message); }
const char* error_text() noexcept { return last_error; }

std::vector<Tensor> tensor_list(const dl_tensor* const* tensors, std::size_t count) {
    std::vector<Tensor> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(tensors[index]->value);
    }
    return result;
}

std::filesystem::path utf8_path(const char* text) {
    return std::filesystem::path(std::u8string(text, text + std::strlen(text)));
}

} // namespace dl::capi

extern "C" {

const char* dl_version(void) { return DL_VERSION; }
const char* dl_last_error(void) { return dl::capi::error_text(); }

dl_tensor_options dl_default_tensor_options(void) {
    const dl::TensorOptions value;
    return {static_cast<dl_dtype>(value.dtype), dl::capi::convert(value.device), value.requires_grad};
}
dl_conv2d_options dl_default_conv2d_options(void) {
    const dl::Conv2dOptions value;
    return {{value.stride.height, value.stride.width}, {value.padding.height, value.padding.width},
        {value.dilation.height, value.dilation.width}, value.groups};
}
dl_pool2d_options dl_default_pool2d_options(void) {
    const dl::Pool2dOptions value;
    return {{value.kernel.height, value.kernel.width}, {value.stride.height, value.stride.width},
        {value.padding.height, value.padding.width}};
}
dl_batch_norm_options dl_default_batch_norm_options(void) {
    const dl::BatchNormOptions value;
    return {value.epsilon, value.momentum, value.affine, value.track_running_stats};
}
dl_attention_options dl_default_attention_options(void) {
    const dl::AttentionOptions value;
    return {value.dropout, value.bias};
}
dl_attention_compute_options dl_default_attention_compute_options(void) {
    const dl::AttentionComputeOptions value;
    return {value.dropout, value.causal, static_cast<dl_attention_backend>(value.backend)};
}
dl_sgd_options dl_default_sgd_options(void) {
    const dl::optim::SGDOptions value;
    return {value.learning_rate, value.momentum, value.dampening, value.weight_decay, value.nesterov};
}
dl_adamw_options dl_default_adamw_options(void) {
    const dl::optim::AdamWOptions value;
    return {value.learning_rate, value.beta1, value.beta2, value.epsilon, value.weight_decay};
}
dl_loader_options dl_default_loader_options(void) {
    const dl::data::LoaderOptions value;
    return {value.batch_size, value.shuffle, value.drop_last};
}

} // extern "C"
