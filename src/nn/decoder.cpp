#include "dl/decoder.h"
#include "dl/autograd.h"
#include "dl/amp.h"

#include <cmath>
#include <stdexcept>

namespace dl::nn {

SwiGLU::SwiGLU(Index dimensions, Index hidden, bool bias, TensorOptions options) {
    gate_ = add_module<Linear>("gate", dimensions, hidden, bias, options);
    up_ = add_module<Linear>("up", dimensions, hidden, bias, options);
    down_ = add_module<Linear>("down", hidden, dimensions, bias, options);
}
Tensor SwiGLU::forward(const Tensor& input) { return (*down_)(silu((*gate_)(input)) * (*up_)(input)); }

LoRALinear::LoRALinear(const Linear& base, Index rank, double alpha)
    : weight_(register_parameter("weight", base.weight().detach().clone())), scale_(alpha / static_cast<double>(rank)) {
    if (rank <= 0) throw std::invalid_argument("LoRA rank must be positive");
    if (base.bias()) bias_ = register_parameter("bias", base.bias()->detach().clone());
    const TensorOptions options{.dtype = weight_.dtype(), .device = weight_.device(), .requires_grad = true};
    down_ = add_module<Linear>("down", weight_.size(1), rank, false, options);
    up_ = add_module<Linear>("up", zeros({weight_.size(0), rank}, options));
}
Tensor LoRALinear::forward(const Tensor& input) {
    return linear(input, bound(weight_), bound(bias_)) + (*up_)((*down_)(input)) * scale_;
}
std::shared_ptr<Linear> LoRALinear::merged() const {
    NoGradGuard guard;
    AutocastGuard precision(weight_.device().type, DType::bf16, false);
    auto weight = weight_ + matmul(up_->weight(), down_->weight()) * scale_;
    if (weight.dtype() != weight_.dtype()) weight = weight.astype(weight_.dtype());
    return std::make_shared<Linear>(std::move(weight), bias_ ? std::optional{bias_->clone()} : std::nullopt);
}

namespace {
Tensor rotary_angles(Index dimensions, Index capacity, double base, TensorOptions options) {
    if (dimensions <= 0 || dimensions % 2 != 0 || capacity <= 0 || base <= 0)
        throw std::invalid_argument("RoPE needs an even positive dimension, positive capacity and base");
    NoGradGuard guard;
    options.dtype = accumulation_dtype(options.dtype);
    options.requires_grad = false;
    auto frequencies = exp(arange(static_cast<double>(dimensions / 2), options) * (-2.0 * std::log(base) / dimensions));
    return arange(static_cast<double>(capacity), options).unsqueeze(1) * frequencies;
}
} // namespace

RotaryEmbedding::RotaryEmbedding(Index dimensions, Index capacity, double base, TensorOptions options)
    : RotaryEmbedding(rotary_angles(dimensions, capacity, base, options)) {}
RotaryEmbedding::RotaryEmbedding(const Tensor& angles)
    : cosine_(register_buffer("cosine", cos(angles))), sine_(register_buffer("sine", sin(angles))) {}
Tensor RotaryEmbedding::forward(const Tensor& input, Index offset) {
    const auto& cosine = bound_buffer(cosine_);
    if (offset < 0 || input.size(-2) > cosine.size(0) - offset)
        throw std::out_of_range("RoPE positions exceed the table capacity");
    return rotary_embedding(input, cosine.narrow(0, offset, input.size(-2)),
        bound_buffer(sine_).narrow(0, offset, input.size(-2)));
}

GroupedQueryAttention::GroupedQueryAttention(Index dimensions, Index heads, Index kv_heads, Index context,
                                               TensorOptions options)
    : dimensions_(dimensions), heads_(heads), kv_heads_(kv_heads) {
    if (heads <= 0 || kv_heads <= 0 || dimensions <= 0 || dimensions % heads || heads % kv_heads)
        throw std::invalid_argument("attention dimensions and query/key head counts must divide evenly");
    query_ = add_module<Linear>("query", dimensions, dimensions, false, options);
    key_ = add_module<Linear>("key", dimensions, kv_heads * (dimensions / heads), false, options);
    value_ = add_module<Linear>("value", dimensions, kv_heads * (dimensions / heads), false, options);
    output_ = add_module<Linear>("output", dimensions, dimensions, false, options);
    rotary_ = add_module<RotaryEmbedding>("rotary", dimensions / heads, context, 10000.0, options);
}
Tensor GroupedQueryAttention::forward(const Tensor& input) { return run(input, nullptr); }
Tensor GroupedQueryAttention::decode(const Tensor& input, KVCache& cache) {
    NoGradGuard guard;
    return run(input, &cache);
}
Tensor GroupedQueryAttention::run(const Tensor& input, KVCache* cache) {
    const auto batch = input.size(0), length = input.size(1), head_size = dimensions_ / heads_;
    const auto offset = cache ? cache->length() : 0;
    auto project = [&](Linear& layer, Index heads) {
        return layer(input).reshape({batch, length, heads, head_size}).transpose(1, 2);
    };
    auto q = rotary_->forward(project(*query_, heads_), offset);
    auto k = rotary_->forward(project(*key_, kv_heads_), offset);
    auto v = project(*value_, kv_heads_);
    if (cache) { cache->append(k, v); k = cache->key(); v = cache->value(); }
    // 新块中的 query 采用绝对位置；逐 token 解码时全部历史 key 均可见。
    std::optional<Tensor> mask;
    if (length > 1) {
        auto keys = arange(static_cast<double>(k.size(2)), {.dtype = DType::i64, .device = input.device()});
        auto queries = arange(static_cast<double>(offset), static_cast<double>(offset + length), 1.0,
            {.dtype = DType::i64, .device = input.device()});
        mask = le(keys, queries.unsqueeze(1));
    }
    auto attended = grouped_query_attention(q, k, v, std::move(mask));
    return (*output_)(attended.transpose(1, 2).reshape({batch, length, dimensions_}));
}

TransformerDecoderLayer::TransformerDecoderLayer(Index dimensions, Index heads, Index kv_heads, Index hidden,
                                                 Index context, TensorOptions options) {
    attention_ = add_module<GroupedQueryAttention>("attention", dimensions, heads, kv_heads, context, options);
    feedforward_ = add_module<SwiGLU>("feedforward", dimensions, hidden, false, options);
    attention_norm_ = add_module<RMSNorm>("attention_norm", Shape{dimensions}, 1e-5, true, options);
    feedforward_norm_ = add_module<RMSNorm>("feedforward_norm", Shape{dimensions}, 1e-5, true, options);
}
Tensor TransformerDecoderLayer::forward(const Tensor& input) {
    auto residual = input + (*attention_)((*attention_norm_)(input));
    return residual + (*feedforward_)((*feedforward_norm_)(residual));
}
Tensor TransformerDecoderLayer::decode(const Tensor& input, KVCache& cache) {
    NoGradGuard guard;
    auto residual = input + attention_->decode((*attention_norm_)(input), cache);
    return residual + (*feedforward_)((*feedforward_norm_)(residual));
}

TransformerDecoder::TransformerDecoder(Index layers, Index dimensions, Index heads, Index kv_heads, Index hidden,
                                       Index context, TensorOptions options) : context_(context) {
    if (layers <= 0) throw std::invalid_argument("decoder requires at least one layer");
    for (Index i = 0; i < layers; ++i)
        layers_.push_back(add_module<TransformerDecoderLayer>("layer" + std::to_string(i), dimensions, heads,
            kv_heads, hidden, context, options));
    normalization_ = add_module<RMSNorm>("normalization", Shape{dimensions}, 1e-5, true, options);
}
Tensor TransformerDecoder::forward(const Tensor& input) {
    auto result = input;
    for (const auto& layer : layers_) result = (*layer)(result);
    return (*normalization_)(result);
}
std::vector<KVCache> TransformerDecoder::make_cache() const {
    std::vector<KVCache> result;
    result.reserve(layers_.size());
    for (std::size_t i = 0; i < layers_.size(); ++i) result.emplace_back(context_);
    return result;
}
Tensor TransformerDecoder::decode(const Tensor& input, std::span<KVCache> cache) {
    if (cache.size() != layers_.size()) throw std::invalid_argument("decoder requires one cache per layer");
    if (!cache.empty()) for (const auto& layer : cache)
        if (layer.length() != cache[0].length()) throw std::invalid_argument("decoder cache positions differ");
    NoGradGuard guard;
    auto result = input;
    for (std::size_t i = 0; i < layers_.size(); ++i) result = layers_[i]->decode(result, cache[i]);
    return (*normalization_)(result);
}

} // namespace dl::nn
