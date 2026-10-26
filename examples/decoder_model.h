#pragma once

#include "project_models.h"

namespace examples {

inline constexpr TextModelOptions decoder_text{128, 64, 2, 4, 0.02};

/// 字节词表和输出层共享权重；中间解码器使用 RoPE、分组查询和 SwiGLU。
class ByteDecoder : public dl::nn::UnaryModule {
public:
    explicit ByteDecoder(TextModelOptions options = decoder_text) {
        auto weight = dl::randn({256, options.dimensions}) * options.embedding_scale;
        weight.requires_grad_(true);
        tokens_ = add_module<dl::nn::Embedding>("tokens", weight);
        decoder_ = add_module<dl::nn::TransformerDecoder>("decoder", options.layers, options.dimensions,
            options.heads, 2, 4 * options.dimensions, options.length);
        output_ = add_module<dl::nn::Linear>("output", tokens_->weight());
    }
    void set_attention_backend(dl::AttentionBackend backend) {
        if (backend != dl::AttentionBackend::math)
            throw std::invalid_argument("grouped-query decoder uses the math attention backend");
    }
    dl::Tensor forward(const dl::Tensor& input) override {
        return (*output_)((*decoder_)((*tokens_)(input)));
    }
    std::vector<dl::nn::KVCache> make_cache() const { return decoder_->make_cache(); }
    dl::Tensor decode(const dl::Tensor& input, std::span<dl::nn::KVCache> cache) {
        dl::NoGradGuard inference;
        return (*output_)(decoder_->decode((*tokens_)(input), cache));
    }
private:
    std::shared_ptr<dl::nn::Embedding> tokens_;
    std::shared_ptr<dl::nn::TransformerDecoder> decoder_;
    std::shared_ptr<dl::nn::Linear> output_;
};

} // namespace examples
