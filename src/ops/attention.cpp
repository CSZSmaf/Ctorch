#include "dl/ops.h"

#include "autograd/record.h"
#include "core/tensor_impl.h"
#include "dl/random.h"
#include "ops/cudnn_attention.h"
#include "ops/grouped_attention.h"
#include "ops/numeric.h"

#ifdef CTORCH_USE_CUDA
#include "kernels/cuda/ops.h"
#endif

#include <algorithm>
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

Tensor causal_window(Index first, Index length, Index source, Device device) {
    TensorOptions positions{.dtype = DType::i64, .device = device};
    auto keys = arange(static_cast<double>(source), positions);
    auto queries = arange(static_cast<double>(first), static_cast<double>(first + length), 1.0, positions);
    return le(keys, queries.unsqueeze(1));
}

Index query_block_size(const Tensor& query, const Tensor& key) {
    const auto columns = std::max<Index>(1, query.size(0) * query.size(1) * key.size(2));
    // FP32 下每块分数约占 32 MiB。批量、头数或 key 长度增加时减少查询行数。
    return std::max<Index>(1, (8 << 20) / columns);
}

Tensor attention_weights(Tensor scores, double scale, const std::optional<Tensor>& mask,
                         std::optional<Index> first) {
#ifdef CTORCH_USE_CUDA
    if (scores.device().type == DeviceType::cuda && scores.dtype() == DType::f32 && !mask) {
        // scores 来自本次矩阵乘法。它的反向只读取 Q/K，因此可以原地写入概率。
        auto data = detail::data_of(scores);
        const auto factor = static_cast<float>(scale);
        detail::cuda::attention_softmax_inplace(data, factor, first.value_or(-1));
        return detail::record(data, {detail::gradient_edge(scores)},
            [data, factor](const auto& grad, auto& sink, const auto& edges) {
                sink.add_owned(edges[0], detail::cuda::attention_softmax_backward(grad, data, factor));
            });
    }
#endif
    scores = mask_scores(scores * scale, mask);
    const auto source = scores.size(3);
    if (first && *first < source - 1)
        scores = mask_scores(scores, causal_window(*first, scores.size(2), source, scores.device()));
    return softmax(scores, -1);
}

Tensor grouped_block(const Tensor& query, const Tensor& key, const Tensor& value,
                     const std::optional<Tensor>& mask, std::optional<Index> first) {
    const auto batch = query.size(0), heads = query.size(1), length = query.size(2);
    const auto kv_heads = key.size(1), source = key.size(2), rows = heads / kv_heads * length;
    auto q = query.reshape({batch, kv_heads, rows, query.size(3)});
    auto scores = matmul(q, key.transpose(-2, -1)).reshape({batch, heads, length, source});
    auto weights = attention_weights(scores, 1.0 / std::sqrt(static_cast<double>(query.size(3))), mask, first);
    return matmul(weights.reshape({batch, kv_heads, rows, source}), value)
        .reshape({batch, heads, length, value.size(3)});
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
    if (options.dropout == 0 && query.ndim() == 4 && key.ndim() == 4 && value.ndim() == 4
        && query.size(0) == key.size(0) && key.size(0) == value.size(0)
        && key.size(1) == value.size(1) && (key.size(1) == query.size(1) || key.size(1) == 1)
        && query.size(2) > query_block_size(query, key)) {
        const Shape scores{query.size(0), query.size(1), query.size(2), key.size(2)};
        if (!keep_mask || detail::broadcast_shape(scores, keep_mask->shape()) == scores)
            return detail::grouped_attention(query, key, value, keep_mask, options.causal ? std::optional<Index>{0} : std::nullopt);
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
    return detail::grouped_attention(query, key, value, keep_mask, std::nullopt);
}

Tensor detail::grouped_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                 const std::optional<Tensor>& keep_mask, std::optional<Index> causal_offset) {
    if (query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4)
        throw std::invalid_argument("grouped-query attention expects [batch,heads,length,features]");
    const auto batch = query.size(0), heads = query.size(1), kv_heads = key.size(1);
    if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || value.size(1) != kv_heads)
        throw std::invalid_argument("query heads must be divisible by matching key/value heads");
    if (key.size(0) != batch || value.size(0) != batch || query.size(3) != key.size(3) || key.size(2) != value.size(2))
        throw std::invalid_argument("attention batch, key dimension and key/value lengths must match");
    const auto length = query.size(2), source = key.size(2), block = query_block_size(query, key);
    if (length <= block) return grouped_block(query, key, value, keep_mask, causal_offset);

    const auto mask = keep_mask ? std::optional{keep_mask->expand({batch, heads, length, source})} : std::nullopt;
    std::vector<Tensor> outputs;
    for (Index start = 0; start < length; start += block) {
        const auto count = std::min(block, length - start);
        const auto first = causal_offset ? std::optional<Index>{*causal_offset + start} : std::nullopt;
        const auto visible = first ? std::min(source, *first + count) : source;
        auto prefix = [visible, source](const Tensor& t) { return visible == source ? t : t.narrow(2, 0, visible); };
        auto part = mask ? std::optional{mask->narrow(2, start, count).narrow(3, 0, visible)} : std::nullopt;
        // 反向按查询块重算概率。因果块只读取最后一行可见的 key/value 前缀。
        outputs.push_back(checkpoint({query.narrow(2, start, count), prefix(key), prefix(value)},
            [part, first](std::span<const Tensor> inputs) {
                return grouped_block(inputs[0], inputs[1], inputs[2], part, first);
            }, {.preserve_rng_state = false}));
    }
    return concat(outputs, 2);
}

Tensor causal_mask(Index query_length, Index key_length, Device device) {
    return causal_window(0, query_length, key_length, device);
}

} // namespace dl
