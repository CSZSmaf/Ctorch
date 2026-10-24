#pragma once

#include "dl/ops.h"

namespace dl::detail {

struct RecurrentState { Tensor hidden; std::optional<Tensor> cell; };

/// 输入与隐藏投影已包含各自偏置；序列入口可提前计算整段输入投影。
RecurrentState recurrent_step(RecurrentKind kind, const Tensor& projected, const Tensor& recurrent,
                               const Tensor& hidden, std::optional<Tensor> cell);

} // namespace dl::detail
