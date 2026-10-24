#pragma once

#include "dl/autograd.h"

namespace dl::detail {

/// 叶槽读取和显式值选择分别按参数、梯度身份去重，保持首次出现的次序。
std::vector<Tensor> leaf_gradients(std::span<const Tensor> parameters);
std::vector<Tensor> unique_gradients(std::span<const std::optional<Tensor>> gradients);

/// 复制每个有值身份一次，输出保留空槽和重复句柄的对应关系。
Gradients clone_gradients(std::span<const std::optional<Tensor>> gradients);

} // namespace dl::detail
