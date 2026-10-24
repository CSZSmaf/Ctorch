#include "dl/tensor.h"

#include "core/dispatch.h"
#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <cmath>

namespace dl {
namespace {

/// 规则序列在宿主按逻辑顺序生成，目标设备与叶属性沿用统一的张量创建入口。
template <class Function> Tensor generate(Shape shape, TensorOptions options, Function value) {
    auto data = detail::empty_data(std::move(shape), options.dtype);
    const auto count = data.layout.numel();
    detail::visit_storage(options.dtype, [&]<class T> {
        auto* output = data.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            output[index] = static_cast<T>(value(index));
        }
    });
    return detail::make_tensor(detail::numeric::to(data, options.device), options.requires_grad);
}

} // namespace

Tensor full_like(const Tensor& input, double value, bool requires_grad) {
    return full(input.shape(), value, {.dtype = input.dtype(), .device = input.device(), .requires_grad = requires_grad});
}
Tensor zeros_like(const Tensor& input, bool requires_grad) { return full_like(input, 0.0, requires_grad); }
Tensor ones_like(const Tensor& input, bool requires_grad) { return full_like(input, 1.0, requires_grad); }

Tensor arange(double end, TensorOptions options) { return arange(0.0, end, 1.0, options); }
Tensor arange(double start, double end, double step, TensorOptions options) {
    const auto count = static_cast<Index>(std::ceil((end - start) / step));
    return generate({count}, options, [=](Index index) { return start + static_cast<double>(index) * step; });
}

Tensor linspace(double start, double end, Index steps, TensorOptions options) {
    return generate({steps}, options, [=](Index index) {
        if (index == 0) return start;
        if (index == steps - 1) return end;
        const double fraction = static_cast<double>(index) / static_cast<double>(steps - 1);
        return (1.0 - fraction) * start + fraction * end;
    });
}

Tensor eye(Index rows, TensorOptions options) { return eye(rows, rows, options); }
Tensor eye(Index rows, Index columns, TensorOptions options) {
    return generate({rows, columns}, options, [=](Index index) { return index / columns == index % columns; });
}

} // namespace dl
