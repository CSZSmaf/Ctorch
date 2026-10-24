#include "dl/nn.h"

#include "nn/initialization.h"

namespace dl::nn {
Linear::Linear(Index input_features, Index output_features, bool bias, TensorOptions options)
    : weight_(register_parameter("weight",
        detail::uniform_parameter({output_features, input_features}, input_features, options))) {
    if (bias) {
        bias_ = register_parameter("bias", detail::uniform_parameter({output_features}, input_features, options));
    }
}

Linear::Linear(Tensor weight, std::optional<Tensor> bias)
    : weight_(register_parameter("weight", std::move(weight))) {
    if (bias) {
        bias_ = register_parameter("bias", std::move(*bias));
    }
}

Tensor Linear::forward(const Tensor& input) {
    return linear(input, bound(weight_), bound(bias_));
}

Tensor Flatten::forward(const Tensor& input) {
    return flatten(input, first_, last_);
}

void Sequential::append(std::string name, std::shared_ptr<UnaryModule> layer) {
    register_module(std::move(name), layer);
    layers_.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    auto result = input;
    for (const auto& layer : layers_) {
        result = layer->forward(result);
    }
    return result;
}

Embedding::Embedding(Index vocabulary, Index dimensions, TensorOptions options)
    : Embedding(randn({vocabulary, dimensions}, detail::trainable_options(options))) {}

Embedding::Embedding(Tensor weight) : weight_(register_parameter("weight", std::move(weight))) {}

Tensor Embedding::forward(const Tensor& indices) {
    return embedding(indices, bound(weight_));
}

LayerNorm::LayerNorm(Shape normalized_shape, double epsilon, bool affine, TensorOptions options)
    : normalized_shape_(std::move(normalized_shape)), epsilon_(epsilon) {
    if (affine) {
        weight_ = register_parameter("weight", ones(normalized_shape_, detail::trainable_options(options)));
        bias_ = register_parameter("bias", zeros(normalized_shape_, detail::trainable_options(options)));
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    return layer_norm(input, normalized_shape_, bound(weight_), bound(bias_), epsilon_);
}

GroupNorm::GroupNorm(Index groups, Index channels, double epsilon, bool affine, TensorOptions options)
    : groups_(groups), epsilon_(epsilon) {
    if (affine) {
        weight_ = register_parameter("weight", ones({channels}, detail::trainable_options(options)));
        bias_ = register_parameter("bias", zeros({channels}, detail::trainable_options(options)));
    }
}

Tensor GroupNorm::forward(const Tensor& input) {
    return group_norm(input, groups_, bound(weight_), bound(bias_), epsilon_);
}

RMSNorm::RMSNorm(Shape normalized_shape, std::optional<double> epsilon, bool affine, TensorOptions options)
    : normalized_shape_(std::move(normalized_shape)), epsilon_(epsilon) {
    if (affine) weight_ = register_parameter("weight", ones(normalized_shape_, detail::trainable_options(options)));
}

Tensor RMSNorm::forward(const Tensor& input) { return rms_norm(input, normalized_shape_, bound(weight_), epsilon_); }

Tensor Dropout::forward(const Tensor& input) {
    return apply_dropout(input, probability_);
}

} // namespace dl::nn
