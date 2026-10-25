#include "ops/numeric.h"

#include "backends/backend.h"
#include "dl/random.h"

namespace dl::detail::numeric {
namespace {

/// 组合计算按需展开低精度值；已有类型化内核的算子在各自入口直接读取原生存储。
TensorData compute(const TensorData& input) { return cast(input, accumulation_dtype(input.dtype)); }

template <class Function> auto compute_pair(Function function, const TensorData& a, const TensorData& b) {
    if (a.dtype != b.dtype && (is_low_precision(a.dtype) || is_low_precision(b.dtype))) {
        const auto dtype = a.dtype == DType::f64 || b.dtype == DType::f64 ? DType::f64 : DType::f32;
        return function(cast(a, dtype), cast(b, dtype));
    }
    return function(a, b);
}

template <class Value, class DeviceOf, class Submit>
void submit_by_device(std::span<Value> values, DeviceOf device_of, Submit submit) {
    while (!values.empty()) {
        const auto device = device_of(values.front());
        std::size_t end = 1;
        while (end < values.size() && device_of(values[end]) == device) ++end;
        submit(backend(device), values.first(end));
        values = values.subspan(end);
    }
}

template <class Update, class Submit>
void submit_updates(std::span<const Update> updates, Submit submit) {
    submit_by_device(updates, [](const auto& update) { return update.parameter->device(); }, submit);
}

} // namespace

TensorData random(Shape shape, Random distribution, double first, double second,
                  TensorOptions options, Generator& generator) {
    if (is_low_precision(options.dtype)) {
        const auto dtype = options.dtype;
        options.dtype = DType::f32;
        return cast(backend(options.device).random(std::move(shape), distribution, first, second, options, generator), dtype);
    }
    return backend(options.device).random(std::move(shape), distribution, first, second, options, generator);
}
DropoutResult dropout(const TensorData& input, double probability, Generator& generator) {
    return backend(input.device()).dropout(compute(input), probability, generator);
}
TensorData random(Shape shape, Random distribution, double first, double second,
                  TensorOptions options, RandomKey key) {
    if (is_low_precision(options.dtype)) {
        const auto dtype = options.dtype;
        options.dtype = DType::f32;
        return cast(backend(options.device).random_keyed(std::move(shape), distribution, first, second, options, key), dtype);
    }
    return backend(options.device).random_keyed(std::move(shape), distribution, first, second, options, key);
}
DropoutResult dropout(const TensorData& input, double probability, RandomKey key) {
    return backend(input.device()).dropout_keyed(compute(input), probability, key);
}
TensorData masked_scale(const TensorData& input, const TensorData& mask, double scale) {
    return backend(input.device()).masked_scale(compute(input), mask, scale);
}

TensorData full(Shape shape, double value, DType dtype, Device device) {
    return backend(device).full(std::move(shape), value, dtype, device);
}
TensorData clone(const TensorData& input) { return backend(input.device()).clone(input); }
TensorData cast(const TensorData& input, DType dtype) {
    return input.dtype == dtype ? input : backend(input.device()).cast(input, dtype);
}
TensorData cast_add(const TensorData& input, const TensorData& offset) {
    return backend(input.device()).cast_add(input, offset);
}
std::array<TensorData, 2> cast_pair(const TensorData& first, const TensorData& second, DType dtype) {
    if (first.dtype == dtype && second.dtype == dtype) return {first, second};
    const auto paired = backend(first.device()).cast_pair;
    return paired ? paired(first, second, dtype) : std::array{cast(first, dtype), cast(second, dtype)};
}

TensorData to(const TensorData& input, Device device) {
    if (input.device() == device) {
        return input;
    }
    auto result = empty_data(input.layout.sizes, input.dtype, device);
    copy_into(result, input);
    return result;
}

void copy_into(TensorData& target, const TensorData& source) {
    if (target.device() == source.device() && target.address() == source.address()
        && target.layout.strides == source.layout.strides) {
        return;
    }
    if (target.device() == source.device()) {
        backend(target.device()).copy_into(target, source);
        return;
    }
    // 设备传输以连续字节区间进行；视图先按逻辑顺序打包，再写回目标布局。
    auto packed = source.layout.is_contiguous() ? source : clone(source);
    auto dense = target.layout.is_contiguous() ? target
        : empty_data(target.layout.sizes, target.dtype, target.device());
    const auto device = source.device().type == DeviceType::cuda ? source.device() : target.device();
    backend(device).transfer(dense.mutable_address(), dense.device(), packed.address(), packed.device(),
                             static_cast<std::size_t>(source.layout.numel()) * element_size(source.dtype));
    if (!target.layout.is_contiguous()) {
        backend(target.device()).copy_into(target, dense);
    }
}
void axpy(TensorData& target, const TensorData& input, double alpha) {
    backend(target.device()).axpy(target, input, alpha);
}
void add_into(TensorData& target, const TensorData& contribution) { axpy(target, contribution, 1.0); }
TensorData unary(const TensorData& input, Unary operation) {
    return backend(input.device()).unary(input, operation);
}
TensorData unary_backward(const TensorData& grad, const TensorData& saved, Unary operation) {
    auto result = empty_data(saved.layout.sizes, accumulation_dtype(saved.dtype), saved.device());
    unary_backward_into(result, grad, saved, operation, false);
    return result;
}
void unary_backward_into(TensorData& target, const TensorData& grad, const TensorData& saved,
                         Unary operation, bool accumulate) {
    backend(saved.device()).unary_backward_into(target, grad, saved, operation, accumulate);
}
TensorData add(const TensorData& a, const TensorData& b) { return compute_pair(backend(a.device()).add, a, b); }
TensorData add_relu(const TensorData& a, const TensorData& b) {
    if (a.dtype == b.dtype) return backend(a.device()).add_relu(a, b);
    return compute_pair(backend(a.device()).add_relu, a, b);
}
TensorData sub(const TensorData& a, const TensorData& b) { return compute_pair(backend(a.device()).sub, a, b); }
TensorData mul(const TensorData& a, const TensorData& b) { return compute_pair(backend(a.device()).mul, a, b); }
TensorData div(const TensorData& a, const TensorData& b) { return compute_pair(backend(a.device()).div, a, b); }
TensorData pow(const TensorData& a, const TensorData& b) { return compute_pair(backend(a.device()).pow, a, b); }
TensorData scale(const TensorData& input, double factor) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype), input.device());
    scale_into(result, input, factor, false);
    return result;
}
void scale_into(TensorData& target, const TensorData& input, double factor, bool accumulate) {
    backend(input.device()).scale_into(target, input, factor, accumulate);
}
TensorData where(const TensorData& mask, const TensorData& a, const TensorData& b) {
    if (a.dtype != b.dtype && (is_low_precision(a.dtype) || is_low_precision(b.dtype))) {
        return compute_pair([&](const auto& left, const auto& right) { return backend(a.device()).where(mask, left, right); }, a, b);
    }
    return backend(a.device()).where(mask, a, b);
}
TensorData compare(const TensorData& a, const TensorData& b, Compare operation) {
    return compute_pair([&](const auto& left, const auto& right) {
        return backend(a.device()).compare(compute(left), compute(right), operation);
    }, a, b);
}
TensorData sum(const TensorData& input, const Shape& axes, KeepDim keep) {
    return backend(input.device()).sum(compute(input), axes, keep);
}
void sum_into(TensorData& target, const TensorData& input, const Shape& axes, bool accumulate) {
    backend(input.device()).sum_into(target, compute(input), axes, accumulate);
}
TensorData amax(const TensorData& input, const Shape& axes, KeepDim keep) {
    return backend(input.device()).amax(compute(input), axes, keep);
}
TensorData arg_extreme(const TensorData& input, Index axis, KeepDim keep, bool maximum) {
    return backend(input.device()).arg_extreme(compute(input), axis, keep, maximum);
}
Shape matmul_shape(const TensorData& a, const TensorData& b) {
    const auto left = std::span<const Index>(a.layout.sizes), right = std::span<const Index>(b.layout.sizes);
    auto shape = broadcast_shape(left.first(left.size() - 2), right.first(right.size() - 2));
    shape.insert(shape.end(), {left[left.size() - 2], right.back()});
    return shape;
}
TensorData matmul(const TensorData& a, const TensorData& b) {
    auto result = empty_data(matmul_shape(a, b), accumulation_dtype(a.dtype), a.device());
    matmul_into(result, a, b, false);
    return result;
}
void matmul_into(TensorData& target, const TensorData& a, const TensorData& b, bool accumulate) {
    backend(a.device()).matmul_into(target, a, b, accumulate);
}
TensorData linear(const TensorData& input, const TensorData& weight,
                  const std::optional<TensorData>& bias, Shape output_shape) {
    auto result = empty_data(std::move(output_shape), accumulation_dtype(input.dtype), input.device());
    const auto offset = bias ? std::optional{cast(*bias, result.dtype)} : std::nullopt;
    const auto fused = backend(input.device()).linear_into;
    if (offset && fused && fused(result, input, weight, *offset)) return result;
    auto matrix = reshape(result, {input.layout.sizes[0], weight.layout.sizes[0]});
    matmul_into(matrix, input, transpose(weight, 0, 1), false);
    if (offset) add_into(result, expand(*offset, result.layout.sizes));
    return result;
}
TensorData softmax(const TensorData& input, Index axis, bool logarithmic) {
    return backend(input.device()).softmax(input, axis, logarithmic);
}
TensorData softmax_backward(const TensorData& grad, const TensorData& output, Index axis, bool logarithmic) {
    return backend(output.device()).softmax_backward(grad, output, axis, logarithmic);
}

NormalizationResult normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon, NormalizationKind kind,
    bool save_statistics) {
    const auto dtype = accumulation_dtype(input.dtype);
    NormalizationResult result{empty_data(input.layout.sizes, dtype, input.device()), std::nullopt};
    if (save_statistics) result.statistics = empty_data({2, input.layout.numel() / width}, dtype, input.device());
    backend(input.device()).normalization(input, weight, bias, width, epsilon, kind, result);
    return result;
}

void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    Index width, NormalizationKind kind, NormalizationGradients& targets) {
    backend(input.device()).normalization_backward(grad, input, statistics, weight,
        width, kind, targets);
}

NormalizationResult batch_normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, std::optional<BatchNormalizationState> state,
    bool training, double momentum, double epsilon) {
    const auto dtype = accumulation_dtype(input.dtype);
    NormalizationResult result{empty_data(input.layout.sizes, dtype, input.device()),
        empty_data({2, input.layout.sizes[1]}, dtype, input.device())};
    backend(input.device()).batch_normalization(input, weight, bias, state, training, momentum, epsilon, result);
    return result;
}

void batch_normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    bool batch_statistics, NormalizationGradients& targets) {
    backend(input.device()).batch_normalization_backward(grad, input, statistics, weight, batch_statistics, targets);
}

TensorData gather(const TensorData& input, Index axis, const TensorData& indices) {
    return backend(input.device()).gather(input, axis, indices);
}
TensorData repeat(const TensorData& input, Shape shape) {
    return backend(input.device()).repeat(input, std::move(shape));
}
void repeat_backward_into(TensorData& target, const TensorData& grad) {
    backend(grad.device()).repeat_backward_into(target, grad);
}
void scatter_add_into(TensorData& target, Index axis, const TensorData& indices, const TensorData& source) {
    backend(target.device()).scatter_add_into(target, axis, indices, source);
}
CrossEntropyResult cross_entropy(const TensorData& logits, const TensorData& labels,
    LossReduction reduction, std::optional<Index> ignore_index, bool save_statistics) {
    const auto dtype = accumulation_dtype(logits.dtype);
    CrossEntropyResult result{empty_data(reduction == LossReduction::none ? labels.layout.sizes : Shape{},
        dtype, logits.device()), std::nullopt};
    if (save_statistics) result.statistics = empty_data({2 * labels.layout.numel() + 1}, dtype, logits.device());
    backend(logits.device()).cross_entropy(logits, labels, reduction, ignore_index, result);
    return result;
}
void cross_entropy_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, const TensorData& statistics,
    LossReduction reduction, std::optional<Index> ignore_index, bool accumulate) {
    backend(logits.device()).cross_entropy_backward_into(target, grad, logits, labels, statistics,
        reduction, ignore_index, accumulate);
}
TensorData binary_logits(const TensorData& logits, const TensorData& labels) {
    return compute_pair(backend(logits.device()).binary_logits, logits, labels);
}
void binary_logits_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, double factor, bool target_gradient, bool accumulate) {
    compute_pair([&](const auto& input, const auto& labels) {
        backend(input.device()).binary_logits_backward_into(target, grad, input, labels,
            factor, target_gradient, accumulate);
    }, logits, labels);
}
void momentum(TensorData& buffer, const TensorData& gradient, double decay, double scale) {
    backend(buffer.device()).momentum(buffer, gradient, decay, scale);
}
TensorData unscale_gradients(std::span<TensorData> gradients, double inverse_scale) {
    return backend(gradients.front().device()).unscale_gradients(gradients, inverse_scale);
}
TensorData clip_grad_norm(std::span<TensorData> gradients, double max_norm, double order) {
    return backend(gradients.front().device()).clip_grad_norm(gradients, max_norm, order);
}
void clip_grad_value(std::span<TensorData> gradients, double bound) {
    submit_by_device(gradients, [](const auto& gradient) { return gradient.device(); },
        [bound](const auto& implementation, auto batch) { implementation.clip_grad_value(batch, bound); });
}
void sgd(std::span<const SGDUpdate> updates) {
    submit_updates(updates, [](const auto& implementation, auto batch) { implementation.sgd(batch); });
}
void adam(std::span<const AdamUpdate> updates) {
    submit_updates(updates, [](const auto& implementation, auto batch) { implementation.adam(batch); });
}

void rmsprop(std::span<const RMSpropUpdate> updates) {
    submit_updates(updates, [](const auto& implementation, auto batch) { implementation.rmsprop(batch); });
}

void adagrad(std::span<const AdagradUpdate> updates) {
    submit_updates(updates, [](const auto& implementation, auto batch) { implementation.adagrad(batch); });
}

TensorData im2col(const TensorData& input, Size2 kernel, const Conv2dOptions& options) {
    return backend(input.device()).im2col(input, kernel, options);
}
void col2im_into(TensorData& target, const TensorData& columns, Size2 kernel,
                 const Conv2dOptions& options, bool accumulate) {
    backend(columns.device()).col2im_into(target, columns, kernel, options, accumulate);
}
MaxPoolResult max_pool2d(const TensorData& input, const Pool2dOptions& options, bool save_indices) {
    return backend(input.device()).max_pool2d(compute(input), options, save_indices);
}
void max_pool2d_backward_into(TensorData& target, const TensorData& grad, const TensorData& indices) {
    backend(grad.device()).max_pool2d_backward_into(target, grad, indices);
}
TensorData avg_pool2d(const TensorData& input, const Pool2dOptions& options, bool count_include_pad) {
    return backend(input.device()).avg_pool2d(compute(input), options, count_include_pad);
}
void avg_pool2d_backward_into(TensorData& target, const TensorData& grad, const Pool2dOptions& options,
                              bool count_include_pad) {
    backend(grad.device()).avg_pool2d_backward_into(target, grad, options, count_include_pad);
}
TensorData adaptive_avg_pool2d(const TensorData& input, Size2 output_size) {
    return backend(input.device()).adaptive_avg_pool2d(compute(input), output_size);
}
void adaptive_avg_pool2d_backward_into(TensorData& target, const TensorData& grad) {
    backend(grad.device()).adaptive_avg_pool2d_backward_into(target, grad);
}
TensorData interpolate(const TensorData& input, Size2 output_size, InterpolationMode mode, bool align_corners) {
    return backend(input.device()).interpolate(compute(input), output_size, mode, align_corners);
}
void interpolate_backward_into(TensorData& target, const TensorData& grad, InterpolationMode mode, bool align_corners) {
    backend(grad.device()).interpolate_backward_into(target, grad, mode, align_corners);
}

} // namespace dl::detail::numeric
