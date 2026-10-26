#include "dl/ops.h"

namespace dl {
namespace {

Tensor reduce(const Tensor& values, LossReduction reduction) {
    if (reduction == LossReduction::none) return values;
    return reduction == LossReduction::sum ? sum(values) : mean(values);
}

} // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& labels, CrossEntropyOptions options) {
    if (!options.weight && options.label_smoothing == 0.0) {
        return options.ignore_index ? cross_entropy(logits, labels, options.reduction, *options.ignore_index)
                                    : cross_entropy(logits, labels, options.reduction);
    }
    const Index axis = logits.ndim() == 1 ? 0 : 1;
    const auto dtype = accumulation_dtype(logits.dtype());
    auto keep = options.ignore_index
        ? ne(labels, from_span<Index>(std::span(&*options.ignore_index, 1), {}, false, labels.device()))
        : ones(labels.shape(), {.dtype = DType::boolean, .device = labels.device()});
    auto selected = where(keep, labels, zeros_like(labels));
    auto probabilities = log_softmax(logits, axis);
    auto losses = -gather(probabilities, axis, selected.unsqueeze(axis)).reshape(labels.shape());
    auto weights = options.weight ? options.weight->detach()
        : ones({logits.size(axis)}, {.dtype = dtype, .device = logits.device()});
    auto selected_weights = index_select(weights, 0, selected.reshape({-1})).reshape(labels.shape());
    losses = selected_weights * losses;
    if (options.label_smoothing != 0.0) {
        Shape broadcast(static_cast<std::size_t>(logits.ndim()), 1);
        broadcast[axis] = logits.size(axis);
        auto smooth = -sum(probabilities * weights.reshape(broadcast), {axis}) / static_cast<double>(logits.size(axis));
        losses = (1.0 - options.label_smoothing) * losses + options.label_smoothing * smooth;
    }
    losses = where(keep, losses, zeros_like(losses));
    if (options.reduction == LossReduction::none) return losses;
    auto total = sum(losses);
    return options.reduction == LossReduction::sum ? total
        : total / sum(where(keep, selected_weights, zeros_like(selected_weights)));
}

Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target, BinaryLossOptions options) {
    if (!options.weight && !options.positive_weight)
        return binary_cross_entropy_with_logits(logits, target, options.reduction);
    auto positive = target * softplus(-logits);
    if (options.positive_weight) positive = positive * options.positive_weight->detach();
    auto losses = (1.0 - target) * softplus(logits) + positive;
    if (options.weight) losses = losses * options.weight->detach();
    return reduce(losses, options.reduction);
}

Tensor focal_loss(const Tensor& logits, const Tensor& target, double gamma, double alpha, LossReduction reduction) {
    auto losses = binary_cross_entropy_with_logits(logits, target, LossReduction::none);
    auto weight = alpha * target + (1.0 - alpha) * (1.0 - target);
    // 二元标签把正确类别的 logit 写成带符号形式，避免从接近一的概率做减法。
    auto modulation = exp(-gamma * softplus((2.0 * target - 1.0) * logits));
    return reduce(weight * modulation * losses, reduction);
}

Tensor dice_loss(const Tensor& probability, const Tensor& target, Shape axes, double epsilon, LossReduction reduction) {
    auto intersection = sum(probability * target, axes);
    auto total = sum(probability + target, std::move(axes));
    return reduce(1.0 - (2.0 * intersection + epsilon) / (total + epsilon), reduction);
}

Tensor kl_div(const Tensor& input, const Tensor& target, LossReduction reduction, bool log_target) {
    if (log_target) return reduce(exp(target) * (target - input), reduction);
    auto positive = gt(target, zeros_like(target));
    auto logarithm = log(where(positive, target, ones_like(target)));
    return reduce(target * (logarithm - input), reduction);
}

} // namespace dl
