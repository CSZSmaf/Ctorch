#include "dl/ops.h"

#include "autograd/record.h"
#include "core/tensor_impl.h"
#include "dl/random.h"
#include "ops/cudnn_attention.h"
#include "ops/numeric.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace dl {

Tensor dropout(const Tensor& input, double probability, bool training) {
    return dropout(input, probability, default_generator(), training);
}

namespace {

template <class Source>
Tensor apply_dropout(const Tensor& input, double probability, Source& source, bool training) {
    if (!training || probability == 0.0) {
        return input;
    }
    if (probability == 1.0) {
        return where(zeros({}, {.dtype = DType::boolean, .device = input.device()}), input, 0.0);
    }
    auto result = detail::numeric::dropout(detail::data_of(input), probability, source);
    return detail::record(std::move(result.output), {detail::gradient_edge(input)},
        [mask = std::move(result.mask), scale = 1.0 / (1.0 - probability)]
        (const auto& grad, auto& sink, const auto& edges) {
            sink.add_owned(edges[0], detail::numeric::masked_scale(grad, mask, scale));
        });
}

Tensor attention_scores(const Tensor& query, const Tensor& key) {
    return matmul(query, key.transpose(-2, -1)) / std::sqrt(static_cast<double>(query.size(-1)));
}

Tensor mask_scores(const Tensor& scores, const std::optional<Tensor>& keep_mask) {
    return keep_mask ? where(*keep_mask, scores, -std::numeric_limits<double>::infinity()) : scores;
}

template <class Source>
Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value,
                 const std::optional<Tensor>& keep_mask, AttentionComputeOptions options, Source& source) {
    if (options.backend == AttentionBackend::cudnn) {
        RandomKey random{0, 0};
        if (static_cast<float>(options.dropout) > 0.0f) {
            if constexpr (std::is_same_v<Source, Generator>) random.seed = source.engine()();
            else random = source;
        }
        return detail::cudnn_attention(query, key, value, keep_mask, options, random);
    }
    auto scores = mask_scores(attention_scores(query, key), keep_mask);
    if (options.causal)
        scores = where(causal_mask(query.size(-2), key.size(-2), query.device()),
            scores, -std::numeric_limits<double>::infinity());
    auto probabilities = apply_dropout(softmax(scores, -1), options.dropout, source, true);
    return matmul(probabilities, value);
}

} // namespace

Tensor dropout(const Tensor& input, double probability, Generator& generator, bool training) {
    return apply_dropout(input, probability, generator, training);
}

Tensor dropout(const Tensor& input, double probability, RandomKey key, bool training) {
    return apply_dropout(input, probability, key, training);
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::optional<Tensor> keep_mask, double dropout_probability) {
    return scaled_dot_product_attention({.dropout = dropout_probability}, query, key, value, std::move(keep_mask));
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::optional<Tensor> keep_mask, double dropout_probability, RandomKey random_key) {
    return scaled_dot_product_attention({.dropout = dropout_probability}, query, key, value,
        random_key, std::move(keep_mask));
}

Tensor scaled_dot_product_attention(AttentionComputeOptions options,
    const Tensor& query, const Tensor& key, const Tensor& value, std::optional<Tensor> keep_mask) {
    return attention(query, key, value, keep_mask, options, default_generator());
}

Tensor scaled_dot_product_attention(AttentionComputeOptions options,
    const Tensor& query, const Tensor& key, const Tensor& value, RandomKey random_key,
    std::optional<Tensor> keep_mask) {
    return attention(query, key, value, keep_mask, options, random_key);
}

Tensor grouped_query_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                std::optional<Tensor> keep_mask) {
    if (query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4)
        throw std::invalid_argument("grouped-query attention expects [batch,heads,length,features]");
    const auto batch = query.size(0), heads = query.size(1), kv_heads = key.size(1);
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || value.size(1) != kv_heads)
        throw std::invalid_argument("query heads must be divisible by matching key/value heads");
    if (key.size(0) != batch || value.size(0) != batch || query.size(3) != key.size(3) || key.size(2) != value.size(2))
        throw std::invalid_argument("attention batch, key dimension and key/value lengths must match");
    const auto groups = heads / kv_heads, length = query.size(2), source = key.size(2);
    // 同一 KV 头服务的查询行合成一块矩阵，反向以同一次矩阵乘法汇总它们的贡献。
    const auto rows = groups * length;
    auto q = query.reshape({batch, kv_heads, rows, query.size(3)});
    auto scores = attention_scores(q, key).reshape({batch, heads, length, source});
    auto weights = softmax(mask_scores(scores, keep_mask), -1).reshape({batch, kv_heads, rows, source});
    return matmul(weights, value).reshape({batch, heads, length, value.size(3)});
}

Tensor causal_mask(Index query_length, Index key_length, Device device) {
    auto host = detail::empty_data({query_length, key_length}, DType::boolean);
    auto* values = host.mutable_data<bool>();
    for (Index query = 0; query < query_length; ++query) {
        for (Index key = 0; key < key_length; ++key) {
            values[query * key_length + key] = key <= query;
        }
    }
    return detail::make_tensor(detail::numeric::to(host, device));
}

} // namespace dl
