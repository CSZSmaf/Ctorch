#pragma once

#include "dl/random.h"

#include <cmath>

namespace dl::detail {

inline TensorOptions trainable_options(TensorOptions options) {
    options.requires_grad = true;
    return options;
}

/// Linear 与 Conv2d 共用按 fan_in 缩放的均匀参数初始化。
inline Tensor uniform_parameter(Shape shape, Index fan_in, TensorOptions options) {
    const double bound = 1.0 / std::sqrt(static_cast<double>(fan_in));
    return uniform(std::move(shape), -bound, bound, trainable_options(options));
}

} // namespace dl::detail
