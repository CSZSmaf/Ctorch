#include "api/common.h"
#include "dl/decoder_c_api.h"

struct dl_decoder_cache { std::vector<dl::nn::KVCache> value; };

using namespace dl::capi;

extern "C" {

dl_status dl_tensordot(const dl_tensor* a, const dl_tensor* b,
    dl_shape axes_a, dl_shape axes_b, dl_tensor** output) {
    return make_handle(output, [&] { return dl::tensordot(a->value, b->value, convert(axes_a), convert(axes_b)); });
}
dl_status dl_rotary_embedding(const dl_tensor* input, const dl_tensor* cosine,
    const dl_tensor* sine, dl_tensor** output) {
    return make_handle(output, [&] { return dl::rotary_embedding(input->value, cosine->value, sine->value); });
}
dl_status dl_grouped_query_attention(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::grouped_query_attention(query->value, key->value, value->value, optional(keep_mask));
    });
}
dl_status dl_swiglu_create(int64_t dimensions, int64_t hidden, int32_t bias,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::SwiGLU>(dimensions, hidden, bias != 0, convert(options)); });
}
dl_status dl_lora_linear_create(const dl_module* base, int64_t rank, double alpha, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::LoRALinear>(
        dynamic_cast<const dl::nn::Linear&>(*base->value), rank, alpha); });
}
dl_status dl_lora_linear_merge(const dl_module* adapter, dl_module** output) {
    return make_handle(output, [&] { return dynamic_cast<const dl::nn::LoRALinear&>(*adapter->value).merged(); });
}
dl_status dl_rotary_embedding_create(int64_t dimensions, int64_t capacity, double base,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::RotaryEmbedding>(dimensions, capacity, base, convert(options)); });
}
dl_status dl_rotary_embedding_forward(dl_module* rotary, const dl_tensor* input, int64_t offset, dl_tensor** output) {
    return make_handle(output, [&] { return dynamic_cast<dl::nn::RotaryEmbedding&>(*rotary->value).forward(input->value, offset); });
}
dl_status dl_grouped_query_attention_create(int64_t dimensions, int64_t heads, int64_t kv_heads,
    int64_t context, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::GroupedQueryAttention>(
        dimensions, heads, kv_heads, context, convert(options)); });
}
dl_status dl_transformer_decoder_create(int64_t layers, int64_t dimensions, int64_t heads,
    int64_t kv_heads, int64_t hidden, int64_t context, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] { return std::make_shared<dl::nn::TransformerDecoder>(
        layers, dimensions, heads, kv_heads, hidden, context, convert(options)); });
}
dl_status dl_decoder_cache_create(const dl_module* decoder, dl_decoder_cache** output) {
    return make_handle(output, [&] { return dynamic_cast<const dl::nn::TransformerDecoder&>(*decoder->value).make_cache(); });
}
void dl_decoder_cache_release(dl_decoder_cache* cache) { delete cache; }
dl_status dl_decoder_cache_clear(dl_decoder_cache* cache) {
    return invoke([&] { for (auto& layer : cache->value) layer.clear(); });
}
dl_status dl_decoder_cache_truncate(dl_decoder_cache* cache, int64_t length) {
    return invoke([&] { for (auto& layer : cache->value) layer.truncate(length); });
}
dl_status dl_decoder_cache_length(const dl_decoder_cache* cache, int64_t* length) {
    return invoke([&] { *length = cache->value.empty() ? 0 : cache->value.front().length(); });
}
dl_status dl_transformer_decoder_decode(dl_module* decoder, const dl_tensor* input,
    dl_decoder_cache* cache, dl_tensor** output) {
    return make_handle(output, [&] {
        return dynamic_cast<dl::nn::TransformerDecoder&>(*decoder->value).decode(input->value, cache->value);
    });
}

} // extern "C"
