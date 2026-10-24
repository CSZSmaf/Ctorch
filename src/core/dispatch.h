#pragma once

#include "dl/types.h"

#include <utility>
#include <type_traits>

namespace dl::detail {

/// FP16/BF16 的逐元素计算和归约使用 float，CPU double 保持自己的累积精度。
template <class T> using FloatAccumulator = std::conditional_t<std::is_same_v<T, double>, double, float>;

/// dtype 在进入数值循环前解析，循环中的地址与运算具有具体 C++ 类型。
template <class Function> decltype(auto) visit_dtype(DType dtype, Function&& function) {
    if (dtype == DType::f32) {
        return function.template operator()<float>();
    }
    if (dtype == DType::f64) {
        return function.template operator()<double>();
    }
    if (dtype == DType::i64) {
        return function.template operator()<Index>();
    }
    return function.template operator()<bool>();
}

template <class Function> decltype(auto) visit_floating(DType dtype, Function&& function) {
    if (dtype == DType::f32) {
        return function.template operator()<float>();
    }
    return function.template operator()<double>();
}

/// 存储访问包含低精度值；计算循环仍按各算子的累积类型选择。
template <class Function> decltype(auto) visit_storage(DType dtype, Function&& function) {
    if (dtype == DType::f16) return function.template operator()<Float16>();
    if (dtype == DType::bf16) return function.template operator()<BFloat16>();
    return visit_dtype(dtype, std::forward<Function>(function));
}

} // namespace dl::detail
