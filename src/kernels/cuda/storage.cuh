#pragma once

#include "core/dispatch.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <type_traits>

namespace dl::detail::cuda {

/// 设备存储使用 CUDA 的原生 16 位类型，与公开 Float16/BFloat16 具有相同位表示。
template <class Function> decltype(auto) visit_storage(DType dtype, Function&& function) {
    if (dtype == DType::f16) return function.template operator()<__half>();
    if (dtype == DType::bf16) return function.template operator()<__nv_bfloat16>();
    return visit_dtype(dtype, std::forward<Function>(function));
}

template <class T> constexpr bool low_storage = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

template <class To, class From> __host__ __device__ To storage_cast(From value) {
    if constexpr (std::is_same_v<To, From>) return value;
    else if constexpr (low_storage<To> || low_storage<From>) return static_cast<To>(static_cast<float>(value));
    else return static_cast<To>(value);
}

} // namespace dl::detail::cuda
