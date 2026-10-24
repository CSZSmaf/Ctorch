#include "dl/nn.h"

#include "core/tensor_impl.h"
#include "nn/initialization.h"
#include "ops/numeric.h"

namespace dl::nn {

Conv1d::Conv1d(Index input_channels, Index output_channels, Index kernel,
                Conv1dOptions convolution, bool bias, TensorOptions options)
    : weight_(register_parameter("weight", detail::uniform_parameter(
        {output_channels, input_channels / convolution.groups, kernel},
        input_channels / convolution.groups * kernel, options))), options_(convolution) {
    if (bias) {
        bias_ = register_parameter("bias", detail::uniform_parameter(
            {output_channels}, input_channels / convolution.groups * kernel, options));
    }
}

Conv1d::Conv1d(Tensor weight, std::optional<Tensor> bias, Conv1dOptions convolution)
    : weight_(register_parameter("weight", std::move(weight))), options_(convolution) {
    if (bias) bias_ = register_parameter("bias", std::move(*bias));
}

Tensor Conv1d::forward(const Tensor& input) { return conv1d(input, bound(weight_), bound(bias_), options_); }

Conv2d::Conv2d(Index input_channels, Index output_channels, Size2 kernel,
                Conv2dOptions convolution, bool bias, TensorOptions options)
    : weight_(register_parameter("weight", detail::uniform_parameter(
        {output_channels, input_channels / convolution.groups, kernel.height, kernel.width},
        input_channels / convolution.groups * kernel.height * kernel.width, options))),
      options_(convolution) {
    if (bias) {
        const auto fan_in = input_channels / convolution.groups * kernel.height * kernel.width;
        bias_ = register_parameter("bias", detail::uniform_parameter({output_channels}, fan_in, options));
    }
}

Conv2d::Conv2d(Tensor weight, std::optional<Tensor> bias, Conv2dOptions convolution)
    : weight_(register_parameter("weight", std::move(weight))), options_(convolution) {
    if (bias) {
        bias_ = register_parameter("bias", std::move(*bias));
    }
}

Tensor Conv2d::forward(const Tensor& input) { return conv2d(input, bound(weight_), bound(bias_), options_); }

BatchNorm::BatchNorm(Index channels, BatchNormOptions normalization, TensorOptions options)
    : channels_(channels), options_(normalization) {
    if (options_.affine) {
        weight_ = register_parameter("weight", ones({channels}, detail::trainable_options(options)));
        bias_ = register_parameter("bias", zeros({channels}, detail::trainable_options(options)));
    }
    if (options_.track_running_stats) {
        options.requires_grad = false;
        options.dtype = accumulation_dtype(options.dtype);
        running_mean_ = register_buffer("running_mean", zeros({channels}, options));
        running_variance_ = register_buffer("running_variance", ones({channels}, options));
    }
}

Tensor BatchNorm::forward(const Tensor& input) {
    if (options_.track_running_stats) {
        return batch_norm_(input, {bound_buffer(*running_mean_), bound_buffer(*running_variance_)},
            bound(weight_), bound(bias_), is_training(),
            options_.momentum, options_.epsilon);
    }
    return batch_norm(input, std::nullopt, bound(weight_), bound(bias_), is_training(),
        options_.momentum, options_.epsilon).output;
}

void BatchNorm::reset_running_stats() {
    if (options_.track_running_stats) {
        auto& center = detail::TensorAccess::mutable_data(*running_mean_);
        auto& variance = detail::TensorAccess::mutable_data(*running_variance_);
        detail::numeric::copy_into(center, detail::numeric::full({channels_}, 0.0, center.dtype, center.device()));
        detail::numeric::copy_into(variance, detail::numeric::full({channels_}, 1.0, variance.dtype, variance.device()));
    }
}

} // namespace dl::nn
