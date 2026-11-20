#pragma once

#include "dl/tensor.h"

namespace dl::detail {

/// causal_offset 是当前块第一个查询在 key 序列中的位置。
Tensor grouped_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                         const std::optional<Tensor>& keep_mask, std::optional<Index> causal_offset);

} // namespace dl::detail
