#include "dl/nn.h"

namespace dl::nn {

MultiHeadAttention::MultiHeadAttention(Index dimensions, Index heads, AttentionOptions attention,
                                       TensorOptions options)
    : dimensions_(dimensions), heads_(heads), compute_{.dropout = attention.dropout} {
    query_ = add_module<Linear>("query", dimensions, dimensions, attention.bias, options);
    key_ = add_module<Linear>("key", dimensions, dimensions, attention.bias, options);
    value_ = add_module<Linear>("value", dimensions, dimensions, attention.bias, options);
    output_ = add_module<Linear>("output", dimensions, dimensions, attention.bias, options);
}

void MultiHeadAttention::set_attention_backend(AttentionBackend backend, bool causal) {
    compute_.backend = backend;
    compute_.causal = causal;
}

Tensor MultiHeadAttention::forward(const Tensor& input) { return forward(input, std::nullopt); }
Tensor MultiHeadAttention::forward(const Tensor& input, std::optional<Tensor> keep_mask) {
    return forward(input, input, input, std::move(keep_mask));
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::optional<Tensor> keep_mask) {
    const auto head_size = dimensions_ / heads_;
    auto project = [&](Linear& layer, const Tensor& input) {
        return layer(input).reshape({input.size(0), input.size(1), heads_, head_size}).transpose(1, 2);
    };
    auto q = project(*query_, query);
    auto k = project(*key_, key);
    auto v = project(*value_, value);
    auto compute = compute_;
    if (!is_training()) compute.dropout = 0.0;
    const auto random_key = next_random_key();
    auto attended = random_key
        ? scaled_dot_product_attention(compute, q, k, v, *random_key, std::move(keep_mask))
        : scaled_dot_product_attention(compute, q, k, v, std::move(keep_mask));
    auto joined = attended.transpose(1, 2).reshape({query.size(0), query.size(1), dimensions_});
    return (*output_)(joined);
}

TransformerEncoderLayer::TransformerEncoderLayer(Index dimensions, Index heads, Index feedforward,
                                                   double dropout_probability, TensorOptions options) {
    attention_ = add_module<MultiHeadAttention>("attention", dimensions, heads,
        AttentionOptions{.dropout = dropout_probability}, options);
    attention_norm_ = add_module<LayerNorm>("attention_norm", Shape{dimensions}, 1e-5, true, options);
    feedforward_norm_ = add_module<LayerNorm>("feedforward_norm", Shape{dimensions}, 1e-5, true, options);
    expand_ = add_module<Linear>("expand", dimensions, feedforward, true, options);
    contract_ = add_module<Linear>("contract", feedforward, dimensions, true, options);
    attention_dropout_ = add_module<Dropout>("attention_dropout", dropout_probability);
    hidden_dropout_ = add_module<Dropout>("hidden_dropout", dropout_probability);
    output_dropout_ = add_module<Dropout>("output_dropout", dropout_probability);
}

Tensor TransformerEncoderLayer::forward(const Tensor& input) { return forward(input, std::nullopt); }
void TransformerEncoderLayer::set_attention_backend(AttentionBackend backend, bool causal) {
    attention_->set_attention_backend(backend, causal);
}

Tensor TransformerEncoderLayer::forward(const Tensor& input, std::optional<Tensor> keep_mask) {
    auto normalized = (*attention_norm_)(input);
    auto residual = input + (*attention_dropout_)(attention_->forward(normalized, std::move(keep_mask)));
    auto hidden = (*hidden_dropout_)(gelu((*expand_)((*feedforward_norm_)(residual))));
    return residual + (*output_dropout_)((*contract_)(hidden));
}

TransformerEncoder::TransformerEncoder(Index layers, Index dimensions, Index heads, Index feedforward,
                                         double dropout_probability, TensorOptions options) {
    for (Index layer = 0; layer < layers; ++layer) {
        layers_.push_back(add_module<TransformerEncoderLayer>("layer" + std::to_string(layer),
            dimensions, heads, feedforward, dropout_probability, options));
    }
    normalization_ = add_module<LayerNorm>("normalization", Shape{dimensions}, 1e-5, true, options);
}

Tensor TransformerEncoder::forward(const Tensor& input) { return forward(input, std::nullopt); }
void TransformerEncoder::set_attention_backend(AttentionBackend backend, bool causal) {
    for (const auto& layer : layers_) layer->set_attention_backend(backend, causal);
}

Tensor TransformerEncoder::forward(const Tensor& input, std::optional<Tensor> keep_mask) {
    auto hidden = input;
    for (const auto& layer : layers_) {
        hidden = layer->forward(hidden, keep_mask);
    }
    return (*normalization_)(hidden);
}

} // namespace dl::nn
