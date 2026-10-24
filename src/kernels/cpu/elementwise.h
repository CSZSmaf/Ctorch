#pragma once

#include "core/iteration.h"

namespace dl::detail::cpu {

/// 存储类型与结果类型独立指定，低精度值在调用标量函数时转换，无需整张临时输入。
template <class From, class To = From, class Function>
void unary(const TensorData& input, TensorData& output, Function function) {
    const auto* source = input.data<From>();
    auto* target = output.mutable_data<To>();
    IterationPlan<1> plan(input.layout.sizes, {input.layout.strides});
    plan.for_each([&](Index index, const auto& offsets) {
        target[index] = static_cast<To>(function(source[offsets[0]]));
    });
}

/// 二元遍历也用于 VJP，使宽精度梯度与低精度保存值共用既有布局规划。
template <class Left, class Right = Left, class To = Left, class Function>
void binary(const TensorData& a, const TensorData& b, TensorData& output, Function function) {
    const auto* left = a.data<Left>();
    const auto* right = b.data<Right>();
    auto* target = output.mutable_data<To>();
    const auto& shape = output.layout.sizes;
    IterationPlan<2> plan(shape, {broadcast_strides(a.layout, shape),
                                 broadcast_strides(b.layout, shape)});
    plan.for_each([&](Index index, const auto& offsets) {
        target[index] = static_cast<To>(function(left[offsets[0]], right[offsets[1]]));
    });
}

} // namespace dl::detail::cpu
