#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"

#include <functional>

namespace dl::detail::cpu {

TensorData compare(const TensorData& a, const TensorData& b, numeric::Compare operation) {
    auto shape = broadcast_shape(a.layout.sizes, b.layout.sizes);
    auto result = empty_data(shape, DType::boolean);
    IterationPlan<2> plan(shape, {broadcast_strides(a.layout, shape), broadcast_strides(b.layout, shape)});
    auto* output = result.mutable_data<bool>();
    visit_dtype(a.dtype, [&]<class T> {
        const auto* left = a.data<T>();
        const auto* right = b.data<T>();
        auto apply = [&](auto relation) {
            plan.for_each([&](Index index, const auto& offsets) {
                output[index] = relation(left[offsets[0]], right[offsets[1]]);
            });
        };
        // 运算种类在进入类型化循环前选定，整数比较直接保留原始位宽。
        switch (operation) {
        case numeric::Compare::eq: return apply(std::equal_to<T>{});
        case numeric::Compare::ne: return apply(std::not_equal_to<T>{});
        case numeric::Compare::lt: return apply(std::less<T>{});
        case numeric::Compare::le: return apply(std::less_equal<T>{});
        case numeric::Compare::gt: return apply(std::greater<T>{});
        case numeric::Compare::ge: return apply(std::greater_equal<T>{});
        }
    });
    return result;
}

} // namespace dl::detail::cpu
