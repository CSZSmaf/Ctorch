#pragma once

#include "dl/ops.h"
#include "dl/random.h"

namespace dl::detail {

/// 自定义求导节点与库图之间的入口；key 只表达本次 Dropout 的可重放位置。
Tensor cudnn_attention(const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& keep_mask, AttentionComputeOptions options, RandomKey random);

} // namespace dl::detail
