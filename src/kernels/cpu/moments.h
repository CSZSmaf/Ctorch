#pragma once

#include "dl/types.h"

namespace dl::detail::cpu {

struct Moments { double mean, variance; };

/// CPU 统计使用 double 累加。两遍读取分别求均值和中心平方和，无需逐元素临时存储。
/// RMS 统计使用零中心，读取方式由连续、转置或通道布局的调用方提供。
template <class Read>
Moments moments(Index count, Read read, bool centered = true) {
    double mean = 0;
    if (centered) {
        for (Index i = 0; i < count; ++i) mean += static_cast<double>(read(i));
        mean /= static_cast<double>(count);
    }
    double squares = 0;
    for (Index i = 0; i < count; ++i) {
        const double value = static_cast<double>(read(i)) - mean;
        squares += value * value;
    }
    return {mean, squares / static_cast<double>(count)};
}

} // namespace dl::detail::cpu
