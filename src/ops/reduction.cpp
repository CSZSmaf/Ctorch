#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl::detail::numeric {

Shape reduced_shape(const Shape& shape, const Shape& axes, KeepDim keep) {
    auto result = shape;
    for (auto position = axes.rbegin(); position != axes.rend(); ++position) {
        if (keep == KeepDim::yes) {
            result[*position] = 1;
        } else {
            result.erase(result.begin() + *position);
        }
    }
    return result;
}

namespace {

Shape sum_to_axes(const Shape& input, const Shape& shape) {
    Shape axes;
    const auto leading = input.size() - shape.size();
    for (std::size_t dim = 0; dim < input.size(); ++dim) {
        if (dim < leading || (shape[dim - leading] == 1 && input[dim] != 1)) {
            axes.push_back(static_cast<Index>(dim));
        }
    }
    return axes;
}

} // namespace

TensorData sum_to_shape(const TensorData& input, Shape shape) {
    const auto axes = sum_to_axes(input.layout.sizes, shape);
    auto reduced = axes.empty() ? input : sum(input, axes, KeepDim::yes);
    return reshape(reduced, std::move(shape));
}

void sum_to_shape_into(TensorData& target, const TensorData& input, bool accumulate) {
    const auto axes = sum_to_axes(input.layout.sizes, target.layout.sizes);
    if (!axes.empty()) {
        sum_into(target, input, axes, accumulate);
    } else {
        // 恒等归约按相同逻辑形状读写，输入的切片或广播布局继续由复制/AXPY 处理。
        auto output = reshape(target, input.layout.sizes);
        auto source = cast(input, target.dtype);
        if (accumulate) add_into(output, source);
        else copy_into(output, source);
    }
}

} // namespace dl::detail::numeric

namespace dl {
namespace {

Tensor reduce(const Tensor& input, Shape axes, KeepDim keep, bool average) {
    axes = detail::normalize_axes(std::move(axes), input.ndim());
    // 空轴列表保持逐元素恒等关系，复制数据即可保留数值表示与直接传递梯度。
    if (axes.empty()) return input.clone();
    Index count = 1;
    for (auto axis : axes) {
        count *= input.size(axis);
    }
    const double factor = average ? 1.0 / static_cast<double>(count) : 1.0;
    auto result = detail::numeric::sum(detail::data_of(input), axes, keep);
    if (average) {
        result = detail::numeric::scale(result, factor);
    }
    auto expanded_shape = detail::numeric::reduced_shape(input.shape(), axes, KeepDim::yes);
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [shape = input.shape(), expanded_shape = std::move(expanded_shape), factor, average]
        (const auto& grad, auto& sink, const auto& edges) {
            auto kept = detail::numeric::reshape(grad, expanded_shape);
            auto expanded = detail::numeric::expand(kept, shape);
            if (average) sink.add_owned(edges[0], detail::numeric::scale(expanded, factor));
            else sink.add(edges[0], expanded);
        });
}

} // namespace

Tensor sum(const Tensor& input) { return sum(input, detail::all_axes(input.ndim())); }
Tensor sum(const Tensor& input, Shape axes, KeepDim keep) { return reduce(input, std::move(axes), keep, false); }
Tensor mean(const Tensor& input) { return mean(input, detail::all_axes(input.ndim())); }
Tensor mean(const Tensor& input, Shape axes, KeepDim keep) { return reduce(input, std::move(axes), keep, true); }

Tensor amax(const Tensor& input) { return amax(input, detail::all_axes(input.ndim())); }
Tensor amin(const Tensor& input) { return amin(input, detail::all_axes(input.ndim())); }
Tensor amin(const Tensor& input, Shape axes, KeepDim keep) {
    return axes.empty() ? input.clone() : -amax(-input, std::move(axes), keep);
}

namespace {

Tensor arg_extreme(const Tensor& input, Index axis, KeepDim keep, bool maximum) {
    // 标量视作长度一的轴，索引仍是标量；普通张量直接按其布局归约。
    if (input.ndim() == 0) return zeros({}, {.dtype = DType::i64, .device = input.device()});
    return detail::make_tensor(detail::numeric::arg_extreme(detail::data_of(input),
        detail::axis_index(axis, input.ndim()), keep, maximum));
}

Tensor arg_extreme(const Tensor& input, bool maximum) {
    auto flat = detail::numeric::reshape(detail::data_of(input), {input.numel()});
    return detail::make_tensor(detail::numeric::arg_extreme(flat, 0, KeepDim::no, maximum));
}

} // namespace

Tensor argmax(const Tensor& input) { return arg_extreme(input, true); }
Tensor argmax(const Tensor& input, Index axis, KeepDim keep) { return arg_extreme(input, axis, keep, true); }
Tensor argmin(const Tensor& input) { return arg_extreme(input, false); }
Tensor argmin(const Tensor& input, Index axis, KeepDim keep) { return arg_extreme(input, axis, keep, false); }

Tensor amax(const Tensor& input, Shape axes, KeepDim keep) {
    namespace numeric = detail::numeric;
    axes = detail::normalize_axes(std::move(axes), input.ndim());
    if (axes.empty()) return input.clone();
    auto result = numeric::amax(detail::data_of(input), axes, keep);
    auto maxima = numeric::reshape(result, numeric::reduced_shape(input.shape(), axes, KeepDim::yes));
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [source = detail::data_of(input), maxima = std::move(maxima), axes = std::move(axes)]
        (const auto& grad, auto& sink, const auto& edges) {
            // 相等掩码给出最大位置，按组计数后将上游梯度均分到这些位置。
            auto selected = numeric::cast(numeric::compare(source, maxima, numeric::Compare::eq), source.dtype);
            auto count = numeric::sum(selected, axes, KeepDim::yes);
            auto share = numeric::div(numeric::reshape(grad, maxima.layout.sizes), count);
            sink.add_owned(edges[0], numeric::mul(selected, share));
        });
}

namespace {

Tensor normalized_exponential(const Tensor& input, Index axis, bool logarithmic) {
    axis = detail::axis_index(axis, input.ndim());
    auto result = detail::numeric::softmax(detail::data_of(input), axis, logarithmic);
    auto saved = result;
    return detail::record(std::move(result), {detail::gradient_edge(input)},
        [saved = std::move(saved), axis, logarithmic](const auto& grad, auto& sink, const auto& edges) {
            // 输出已包含反向所需的概率表示，行归约与逐元素 VJP 在同一内核内完成。
            sink.add_owned(edges[0], detail::numeric::softmax_backward(grad, saved, axis, logarithmic));
        });
}

} // namespace

Tensor softmax(const Tensor& input, Index axis) { return normalized_exponential(input, axis, false); }
Tensor log_softmax(const Tensor& input, Index axis) { return normalized_exponential(input, axis, true); }

} // namespace dl
