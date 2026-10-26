#ifndef CTORCH_DECODER_C_API_H
#define CTORCH_DECODER_C_API_H

#include "dl/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 解码器缓存由一个推理请求持有，按层保存 K/V；release 独立于模型生命周期。 */
typedef struct dl_decoder_cache dl_decoder_cache;

DL_API dl_status dl_tensordot(const dl_tensor* a, const dl_tensor* b,
    dl_shape axes_a, dl_shape axes_b, dl_tensor** output);
DL_API dl_status dl_rotary_embedding(const dl_tensor* input, const dl_tensor* cosine,
    const dl_tensor* sine, dl_tensor** output);
DL_API dl_status dl_grouped_query_attention(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_tensor** output);

DL_API dl_status dl_swiglu_create(int64_t dimensions, int64_t hidden, int32_t bias,
    dl_tensor_options options, dl_module** output);
/* base 为 Linear。适配器复制并冻结基层参数；merged 返回独立的 Linear。 */
DL_API dl_status dl_lora_linear_create(const dl_module* base, int64_t rank, double alpha, dl_module** output);
DL_API dl_status dl_lora_linear_merge(const dl_module* adapter, dl_module** output);
DL_API dl_status dl_rotary_embedding_create(int64_t dimensions, int64_t capacity, double base,
    dl_tensor_options options, dl_module** output);
DL_API dl_status dl_rotary_embedding_forward(dl_module* rotary, const dl_tensor* input,
    int64_t offset, dl_tensor** output);
DL_API dl_status dl_grouped_query_attention_create(int64_t dimensions, int64_t heads, int64_t kv_heads,
    int64_t context, dl_tensor_options options, dl_module** output);
DL_API dl_status dl_transformer_decoder_create(int64_t layers, int64_t dimensions, int64_t heads,
    int64_t kv_heads, int64_t hidden, int64_t context, dl_tensor_options options, dl_module** output);

/* 训练使用 dl_module_forward；decode 只处理新增 token，不记录梯度。 */
DL_API dl_status dl_decoder_cache_create(const dl_module* decoder, dl_decoder_cache** output);
DL_API void dl_decoder_cache_release(dl_decoder_cache* cache);
DL_API dl_status dl_decoder_cache_clear(dl_decoder_cache* cache);
DL_API dl_status dl_decoder_cache_truncate(dl_decoder_cache* cache, int64_t length);
DL_API dl_status dl_decoder_cache_length(const dl_decoder_cache* cache, int64_t* length);
DL_API dl_status dl_transformer_decoder_decode(dl_module* decoder, const dl_tensor* input,
    dl_decoder_cache* cache, dl_tensor** output);

#ifdef __cplusplus
}
#endif

#endif
