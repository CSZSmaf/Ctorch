#pragma once

#include "ops/numeric.h"

namespace dl::detail::cpu {

TensorData unscale_gradients(std::span<TensorData> gradients, double inverse_scale);
TensorData clip_grad_norm(std::span<TensorData> gradients, double max_norm, double order);
void clip_grad_value(std::span<TensorData> gradients, double bound);

using numeric::CrossEntropyResult;
using numeric::Unary;

TensorData random(Shape shape, numeric::Random distribution, double first, double second,
                  TensorOptions options, Generator& generator);
numeric::DropoutResult dropout(const TensorData& input, double probability, Generator& generator);
TensorData random_keyed(Shape shape, numeric::Random distribution, double first, double second,
                        TensorOptions options, RandomKey key);
numeric::DropoutResult dropout_keyed(const TensorData& input, double probability, RandomKey key);
TensorData masked_scale(const TensorData& input, const TensorData& mask, double scale);

TensorData full(Shape shape, double value, DType dtype, Device device = Device::cpu());
TensorData clone(const TensorData& input);
TensorData cast(const TensorData& input, DType dtype);
TensorData cast_add(const TensorData& input, const TensorData& offset);
void copy_into(TensorData& target, const TensorData& source);
void axpy(TensorData& target, const TensorData& input, double alpha);
TensorData unary(const TensorData& input, Unary operation);
void unary_backward_into(TensorData& target, const TensorData& grad, const TensorData& saved,
                         Unary operation, bool accumulate);
TensorData add(const TensorData& a, const TensorData& b);
TensorData add_relu(const TensorData& a, const TensorData& b);
TensorData sub(const TensorData& a, const TensorData& b);
TensorData mul(const TensorData& a, const TensorData& b);
TensorData div(const TensorData& a, const TensorData& b);
TensorData pow(const TensorData& a, const TensorData& b);
void scale_into(TensorData& target, const TensorData& input, double factor, bool accumulate);
TensorData where(const TensorData& mask, const TensorData& a, const TensorData& b);
TensorData compare(const TensorData& a, const TensorData& b, numeric::Compare operation);
TensorData sum(const TensorData& input, const Shape& axes, KeepDim keep);
TensorData mean(const TensorData& input, const Shape& axes, KeepDim keep);
numeric::VarianceResult variance(const TensorData& input, const Shape& axes, double correction, KeepDim keep);
void sum_into(TensorData& target, const TensorData& input, const Shape& axes, bool accumulate);
TensorData amax(const TensorData& input, const Shape& axes, KeepDim keep);
TensorData arg_extreme(const TensorData& input, Index axis, KeepDim keep, bool maximum);
TensorData cumsum(const TensorData& input, Index axis, bool reverse);
TensorData argsort(const TensorData& input, Index axis, bool descending);
void matmul_into(TensorData& target, const TensorData& a, const TensorData& b, bool accumulate);
TensorData softmax(const TensorData& input, Index axis, bool logarithmic = false);
TensorData softmax_backward(const TensorData& grad, const TensorData& output, Index axis, bool logarithmic);
void normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon,
    numeric::NormalizationKind kind, numeric::NormalizationResult& result);
void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight, Index width,
    numeric::NormalizationKind kind, numeric::NormalizationGradients& result);
void batch_normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, std::optional<numeric::BatchNormalizationState>& state,
    bool training, double momentum, double epsilon, numeric::NormalizationResult& result);
void batch_normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    bool batch_statistics, numeric::NormalizationGradients& result);
TensorData gather(const TensorData& input, Index axis, const TensorData& indices);
TensorData repeat(const TensorData& input, Shape shape);
void repeat_backward_into(TensorData& target, const TensorData& grad);
void scatter_add_into(TensorData& target, Index axis, const TensorData& indices, const TensorData& source);
void cross_entropy(const TensorData& logits, const TensorData& labels, LossReduction reduction,
    std::optional<Index> ignore_index, CrossEntropyResult& result);
TensorData binary_logits(const TensorData& logits, const TensorData& labels);
void binary_logits_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, double factor, bool target_gradient, bool accumulate);
void cross_entropy_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, const TensorData& statistics,
    LossReduction reduction, std::optional<Index> ignore_index, bool accumulate);
void momentum(TensorData& buffer, const TensorData& gradient, double decay, double scale);
void sgd(std::span<const numeric::SGDUpdate> updates);
void adam(std::span<const numeric::AdamUpdate> updates);
void rmsprop(std::span<const numeric::RMSpropUpdate> updates);
void adagrad(std::span<const numeric::AdagradUpdate> updates);

TensorData im2col(const TensorData& input, Size2 kernel, const Conv2dOptions& options);
void col2im_into(TensorData& target, const TensorData& columns, Size2 kernel,
                 const Conv2dOptions& options, bool accumulate);
numeric::MaxPoolResult max_pool2d(const TensorData& input, const Pool2dOptions& options, bool save_indices);
void max_pool2d_backward_into(TensorData& target, const TensorData& grad, const TensorData& indices);
TensorData avg_pool2d(const TensorData& input, const Pool2dOptions& options, bool count_include_pad);
void avg_pool2d_backward_into(TensorData& target, const TensorData& grad, const Pool2dOptions& options,
                              bool count_include_pad);
TensorData adaptive_avg_pool2d(const TensorData& input, Size2 output_size);
void adaptive_avg_pool2d_backward_into(TensorData& target, const TensorData& grad);
TensorData interpolate(const TensorData& input, Size2 output_size, InterpolationMode mode, bool align_corners);
void interpolate_backward_into(TensorData& target, const TensorData& grad, InterpolationMode mode, bool align_corners);

} // namespace dl::detail::cpu
