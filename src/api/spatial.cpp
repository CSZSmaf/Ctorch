#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_conv1d_options dl_default_conv1d_options(void) {
    const dl::Conv1dOptions value;
    return {value.stride, value.padding, value.dilation, value.groups};
}
dl_pool1d_options dl_default_pool1d_options(void) {
    const dl::Pool1dOptions value;
    return {value.kernel, value.stride, value.padding};
}

dl_status dl_conv1d(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_conv1d_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::conv1d(input->value, weight->value, optional(bias), convert(options)); });
}
dl_status dl_max_pool1d(const dl_tensor* input, dl_pool1d_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::max_pool1d(input->value, convert(options)); });
}
dl_status dl_avg_pool1d(const dl_tensor* input, dl_pool1d_options options, int32_t count_include_pad, dl_tensor** output) {
    return make_handle(output, [&] { return dl::avg_pool1d(input->value, convert(options), count_include_pad != 0); });
}
dl_status dl_adaptive_avg_pool1d(const dl_tensor* input, int64_t output_size, dl_tensor** output) {
    return make_handle(output, [&] { return dl::adaptive_avg_pool1d(input->value, output_size); });
}
dl_status dl_adaptive_avg_pool2d(const dl_tensor* input, dl_size2 output_size, dl_tensor** output) {
    return make_handle(output, [&] { return dl::adaptive_avg_pool2d(input->value, convert(output_size)); });
}
dl_status dl_interpolate(const dl_tensor* input, dl_size2 output_size, dl_interpolation_mode mode,
    int32_t align_corners, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::interpolate(input->value, convert(output_size), convert(mode), align_corners != 0);
    });
}
dl_status dl_batch_norm(const dl_tensor* input, const dl_tensor* running_mean,
    const dl_tensor* running_variance, const dl_tensor* weight, const dl_tensor* bias,
    int32_t training, double momentum, double epsilon, dl_tensor_list** output) {
    return make_handle(output, [&] {
        std::optional<dl::BatchNormState> state;
        if (running_mean) state = dl::BatchNormState{running_mean->value, running_variance->value};
        auto result = dl::batch_norm(input->value, state, optional(weight), optional(bias),
            training != 0, momentum, epsilon);
        std::vector<dl::Tensor> values{std::move(result.output)};
        if (result.state) {
            values.push_back(std::move(result.state->mean));
            values.push_back(std::move(result.state->variance));
        }
        return values;
    });
}

dl_status dl_batch_norm_update(const dl_tensor* input, const dl_tensor* running_mean,
    const dl_tensor* running_variance, const dl_tensor* weight, const dl_tensor* bias,
    int32_t training, double momentum, double epsilon, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::batch_norm_(input->value, {running_mean->value, running_variance->value},
            optional(weight), optional(bias), training != 0, momentum, epsilon);
    });
}

dl_status dl_group_norm(const dl_tensor* input, int64_t groups, const dl_tensor* weight,
    const dl_tensor* bias, double epsilon, dl_tensor** output) {
    return make_handle(output, [&] { return dl::group_norm(input->value, groups, optional(weight), optional(bias), epsilon); });
}
dl_status dl_rms_norm(const dl_tensor* input, dl_shape normalized_shape,
    const dl_tensor* weight, const double* epsilon, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::rms_norm(input->value, convert(normalized_shape), optional(weight),
            epsilon ? std::optional<double>(*epsilon) : std::nullopt);
    });
}

dl_status dl_conv1d_create(int64_t input_channels, int64_t output_channels, int64_t kernel,
    dl_conv1d_options convolution, int32_t bias, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Conv1d>(input_channels, output_channels, kernel,
            convert(convolution), bias != 0, convert(options));
    });
}
dl_status dl_conv1d_from_tensors(const dl_tensor* weight, const dl_tensor* bias,
    dl_conv1d_options convolution, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Conv1d>(weight->value, optional(bias), convert(convolution));
    });
}
dl_status dl_max_pool1d_create(dl_pool1d_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::MaxPool1d>(convert(options)); });
}
dl_status dl_avg_pool1d_create(dl_pool1d_options options, int32_t count_include_pad, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::AvgPool1d>(convert(options), count_include_pad != 0); });
}
dl_status dl_adaptive_avg_pool1d_create(int64_t output_size, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::AdaptiveAvgPool1d>(output_size); });
}
dl_status dl_adaptive_avg_pool2d_create(dl_size2 output_size, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::AdaptiveAvgPool2d>(convert(output_size)); });
}
dl_status dl_upsample_create(dl_size2 output_size, dl_interpolation_mode mode, int32_t align_corners, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::Upsample>(convert(output_size), convert(mode), align_corners != 0);
    });
}
dl_status dl_batch_norm1d_create(int64_t channels, dl_batch_norm_options normalization,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::BatchNorm1d>(channels, convert(normalization), convert(options));
    });
}
dl_status dl_batch_norm1d_reset(dl_module* module) {
    return invoke([&] { static_cast<dl::nn::BatchNorm1d&>(*module->value).reset_running_stats(); });
}
dl_status dl_group_norm_create(int64_t groups, int64_t channels, double epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::GroupNorm>(groups, channels, epsilon, affine != 0, convert(options));
    });
}
dl_status dl_rms_norm_create(dl_shape normalized_shape, const double* epsilon, int32_t affine,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::RMSNorm>(convert(normalized_shape),
            epsilon ? std::optional<double>(*epsilon) : std::nullopt, affine != 0, convert(options));
    });
}

} // extern "C"
