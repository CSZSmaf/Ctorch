#include "dl/ops.h"

#include "autograd/record.h"
#include "core/tensor_impl.h"
#include "dl/random.h"
#include "ops/cudnn_attention.h"
#include "ops/numeric.h"

#include <cmath>
#include <limits>
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
    auto scores = matmul(query, key.transpose(-2, -1)) / std::sqrt(static_cast<double>(query.size(-1)));
    if (keep_mask) scores = where(*keep_mask, scores, -std::numeric_limits<double>::infinity());
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
