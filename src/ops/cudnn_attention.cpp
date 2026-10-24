#include "ops/cudnn_attention.h"

#include "autograd/record.h"
#include "dl/amp.h"
#include "ops/numeric.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef CTORCH_USE_CUDNN
#include "backends/cudnn.h"
#endif

namespace dl::detail {

#ifdef CTORCH_USE_CUDNN
namespace {

TensorData packed(const TensorData& input, DType dtype) {
    auto value = numeric::cast(input, dtype);
    if (!value.layout.is_contiguous() || value.layout.offset != 0) value = numeric::clone(value);
    return value;
}

} // namespace
#endif

Tensor cudnn_attention(const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& keep_mask, AttentionComputeOptions options, RandomKey random) {
#ifdef CTORCH_USE_CUDNN
    const auto dtype = query.dtype() == DType::f32 && is_autocast_enabled(DeviceType::cuda)
        ? autocast_dtype(DeviceType::cuda) : query.dtype();
    std::array inputs{packed(data_of(query), dtype), packed(data_of(key), dtype), packed(data_of(value), dtype)};
    std::optional<TensorData> bias;
    if (keep_mask) {
        // cuDNN 偏置在 batch/head 轴广播，两个序列轴展开为完整尺寸供前向和反向共用。
        auto shape = keep_mask->shape();
        shape.insert(shape.begin(), 4 - shape.size(), 1);
        auto mask = numeric::reshape(data_of(*keep_mask), shape);
        shape[2] = query.size(-2);
        shape[3] = key.size(-2);
        mask = numeric::expand(mask, std::move(shape));
        auto zero = numeric::full({}, 0.0, dtype, query.device());
        auto excluded = numeric::full({}, -std::numeric_limits<double>::infinity(), dtype, query.device());
        bias = numeric::where(mask, zero, excluded);
    }
    std::vector edges{gradient_edge(query), gradient_edge(key), gradient_edge(value)};
    const bool needs_grad = std::any_of(edges.begin(), edges.end(), [](const auto& edge) { return bool(edge.node); });
    auto result = cuda::attention_forward(inputs, bias, options, random, needs_grad);
    auto output = numeric::cast(result.output, DType::f32);
    return record(std::move(output), std::move(edges),
        [inputs = std::move(inputs), bias = std::move(bias), result = std::move(result), options, random]
        (const auto& grad, auto& sink, const auto& input_edges) {
            auto seed = packed(grad, inputs[0].dtype);
            auto gradients = cuda::attention_backward(inputs, result.output, *result.statistics,
                seed, bias, options, random);
            for (std::size_t i = 0; i < gradients.size(); ++i)
                sink.add_owned(input_edges[i], std::move(gradients[i]));
        });
#else
    (void)query; (void)key; (void)value; (void)keep_mask; (void)options; (void)random;
    throw std::runtime_error("cuDNN attention requires CTORCH_USE_CUDNN=ON");
#endif
}

} // namespace dl::detail
