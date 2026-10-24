#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/iteration.h"

namespace dl::detail::cpu {
namespace {

/// 输出坐标对输入轴长取余，前导补入的轴只改变重复次数。
Index repeat_offset(Index linear, const Shape& shape, const Layout& source) {
    const auto leading = shape.size() - source.sizes.size();
    Index result = 0;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        if (axis >= leading) {
            const auto original = axis - leading;
            result += (linear % shape[axis] % source.sizes[original]) * source.strides[original];
        }
        linear /= shape[axis];
    }
    return result;
}

} // namespace

TensorData repeat(const TensorData& input, Shape shape) {
    auto result = empty_data(std::move(shape), input.dtype);
    const auto count = result.layout.numel();
    visit_storage(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* output = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            output[index] = source[repeat_offset(index, result.layout.sizes, input.layout)];
        }
    });
    return result;
}

void repeat_backward_into(TensorData& target, const TensorData& grad) {
    const auto count = grad.layout.numel();
    visit_floating(grad.dtype, [&]<class T> {
        const auto* values = grad.data<T>();
        auto* output = target.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            output[repeat_offset(index, grad.layout.sizes, target.layout)] += values[grad.layout.index_offset(index)];
        }
    });
}

TensorData gather(const TensorData& input, Index axis, const TensorData& indices) {
    auto result = empty_data(indices.layout.sizes, input.dtype);
    auto source_strides = input.layout.strides;
    source_strides[axis] = 0;
    IterationPlan<2> plan(indices.layout.sizes, {std::move(source_strides), indices.layout.strides});
    const auto* selected = indices.data<Index>();
    visit_storage(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* target = result.mutable_data<T>();
        plan.for_each([&](Index linear, const auto& offsets) {
            const auto row = selected[offsets[1]];
            target[linear] = source[offsets[0] + row * input.layout.strides[axis]];
        });
    });
    return result;
}

void scatter_add_into(TensorData& target, Index axis, const TensorData& indices, const TensorData& source) {
    auto target_strides = target.layout.strides;
    target_strides[axis] = 0;
    IterationPlan<3> plan(indices.layout.sizes,
        {source.layout.strides, std::move(target_strides), indices.layout.strides});
    const auto* selected = indices.data<Index>();
    visit_floating(source.dtype, [&]<class T> {
        const auto* values = source.data<T>();
        auto* output = target.mutable_data<T>();
        plan.for_each([&](Index, const auto& offsets) {
            const auto row = selected[offsets[2]];
            output[offsets[1] + row * target.layout.strides[axis]] += values[offsets[0]];
        });
    });
}

} // namespace dl::detail::cpu
