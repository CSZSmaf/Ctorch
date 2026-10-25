#pragma once

#include "core/iteration.h"

#include <type_traits>

namespace dl::detail::cpu {

/// 存储类型与结果类型独立指定，低精度值在调用标量函数时转换，无需整张临时输入。
/// 输出为连续布局，累加模式将计算结果直接加到已有元素。
template <class From, class To = From, class Function>
void unary(const TensorData& input, TensorData& output, Function function, bool accumulate = false) {
    const auto* source = input.data<From>();
    auto* target = output.mutable_data<To>();
    if (input.layout.is_contiguous()) {
        const auto count = input.layout.numel();
        for (Index index = 0; index < count; ++index) {
            const To value = static_cast<To>(function(source[index]));
            target[index] = accumulate ? target[index] + value : value;
        }
        return;
    }
    IterationPlan<1> plan(input.layout.sizes, {input.layout.strides});
    plan.for_each([&](Index index, const auto& offsets) {
        const To value = static_cast<To>(function(source[offsets[0]]));
        target[index] = accumulate ? target[index] + value : value;
    });
}

/// 二元遍历也用于 VJP，使宽精度梯度与低精度保存值共用既有布局规划。
template <class Left, class Right = Left, class To = Left, class Function>
void binary(const TensorData& a, const TensorData& b, TensorData& output, Function function, bool accumulate = false) {
    const auto* left = a.data<Left>();
    const auto* right = b.data<Right>();
    auto* target = output.mutable_data<To>();
    const auto& shape = output.layout.sizes;
    IterationPlan<2> plan(shape, {broadcast_strides(a.layout, shape),
                                 broadcast_strides(b.layout, shape)});
    plan.for_each_row([&](Index base, Index width, const auto& offsets, const auto& steps) {
        const auto* x = left + offsets[0];
        const auto* y = right + offsets[1];
        auto apply = [&](auto sx, auto sy) {
            for (Index column = 0; column < width; ++column) {
                const To value = static_cast<To>(function(x[column * sx], y[column * sy]));
                auto& destination = target[base + column];
                destination = accumulate ? destination + value : value;
            }
        };
        // 连续与标量广播的步长成为编译期常量，编译器可直接生成向量循环。
        constexpr std::integral_constant<Index, 0> broadcast;
        constexpr std::integral_constant<Index, 1> contiguous;
        if (steps[0] == 1 && steps[1] == 1) apply(contiguous, contiguous);
        else if (steps[0] == 0 && steps[1] == 1) apply(broadcast, contiguous);
        else if (steps[0] == 1 && steps[1] == 0) apply(contiguous, broadcast);
        else apply(steps[0], steps[1]);
    });
}

} // namespace dl::detail::cpu
