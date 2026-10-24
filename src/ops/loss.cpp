#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

#include <array>

namespace dl {
namespace {

Tensor reduce_loss(const Tensor& losses, LossReduction reduction) {
    if (reduction == LossReduction::none) return losses;
    return reduction == LossReduction::sum ? sum(losses) : mean(losses);
}

Tensor classification_loss(const Tensor& logits, const Tensor& labels,
    LossReduction reduction, std::optional<Index> ignore_index) {
    namespace numeric = detail::numeric;
    const auto edge = detail::gradient_edge(logits);
    auto result = numeric::cross_entropy(detail::data_of(logits), detail::data_of(labels),
        reduction, ignore_index, bool(edge.node));
    // 原输入和逐行统计足以重算概率，类别轴之外的每个位置对应一项标签。
    return detail::record(std::move(result.loss), {edge},
        [input = detail::data_of(logits), targets = detail::data_of(labels),
         statistics = std::move(result.statistics), reduction, ignore_index]
        (const auto& grad, auto& sink, const auto& edges) {
            if (!sink.needs(edges[0])) return;
            auto target = sink.target(edges[0], input.layout.sizes, accumulation_dtype(input.dtype), input.device());
            numeric::cross_entropy_backward_into(target.value, grad, input, targets, *statistics,
                reduction, ignore_index, target.accumulate);
        });
}

} // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& labels) {
    return classification_loss(logits, labels, LossReduction::mean, std::nullopt);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& labels, LossReduction reduction) {
    return classification_loss(logits, labels, reduction, std::nullopt);
}

Tensor nll_loss(const Tensor& log_probabilities, const Tensor& labels, LossReduction reduction) {
    const Index axis = log_probabilities.ndim() == 1 ? 0 : 1;
    auto selected = gather(log_probabilities, axis, labels.unsqueeze(axis));
    return reduce_loss(-selected.reshape(labels.shape()), reduction);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& labels, LossReduction reduction, Index ignore_index) {
    return classification_loss(logits, labels, reduction, ignore_index);
}

Tensor nll_loss(const Tensor& log_probabilities, const Tensor& labels, LossReduction reduction, Index ignore_index) {
    const std::array ignored_value{ignore_index};
    const auto ignored = from_span<Index>(ignored_value, {}, false, labels.device());
    const auto keep = ne(labels, ignored);
    // 忽略位置先映射到有效索引，随后选择零损失，标签本身无需改变。
    const auto safe_labels = where(keep, labels, zeros_like(labels));
    const auto selected = nll_loss(log_probabilities, safe_labels, LossReduction::none);
    const auto losses = where(keep, selected, zeros_like(selected));
    if (reduction == LossReduction::none) return losses;
    auto total = sum(losses);
    return reduction == LossReduction::sum ? total : total / sum(keep.astype(log_probabilities.dtype()));
}

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    return mse_loss(prediction, target, LossReduction::mean);
}

Tensor mse_loss(const Tensor& prediction, const Tensor& target, LossReduction reduction) {
    auto difference = prediction - target;
    return reduce_loss(difference * difference, reduction);
}

Tensor l1_loss(const Tensor& prediction, const Tensor& target, LossReduction reduction) {
    return reduce_loss(abs(prediction - target), reduction);
}

Tensor smooth_l1_loss(const Tensor& prediction, const Tensor& target, double beta, LossReduction reduction) {
    if (beta == 0.0) return l1_loss(prediction, target, reduction);
    auto difference = prediction - target;
    auto magnitude = abs(difference);
    auto threshold = full({}, beta, {.dtype = prediction.dtype(), .device = prediction.device()});
    auto losses = where(lt(magnitude, threshold), difference * difference / (2.0 * beta), magnitude - 0.5 * beta);
    return reduce_loss(losses, reduction);
}

Tensor huber_loss(const Tensor& prediction, const Tensor& target, double delta, LossReduction reduction) {
    return smooth_l1_loss(prediction, target, delta, reduction) * delta;
}

Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target, LossReduction reduction) {
    // 两个 Softplus 分别表示负类别和正类别代价，无需先生成再取对数的概率。
    auto losses = (1.0 - target) * softplus(logits) + target * softplus(-logits);
    return reduce_loss(losses, reduction);
}

Tensor binary_cross_entropy(const Tensor& probability, const Tensor& target, LossReduction reduction) {
    namespace numeric = detail::numeric;
    const auto& p = detail::data_of(probability);
    const auto& t = detail::data_of(target);
    auto one = numeric::full({}, 1.0, p.dtype, p.device());
    auto floor = numeric::full({}, -100.0, p.dtype, p.device());
    auto clipped_log = [&](const auto& value) {
        return numeric::where(numeric::compare(value, floor, numeric::Compare::lt), floor, value);
    };
    auto log_p = clipped_log(numeric::unary(p, numeric::Unary::log));
    auto log_q = clipped_log(numeric::unary(numeric::scale(p, -1.0), numeric::Unary::log1p));
    auto result = numeric::scale(numeric::add(numeric::mul(t, log_p),
        numeric::mul(numeric::sub(one, t), log_q)), -1.0);
    auto losses = detail::record(std::move(result), {detail::gradient_edge(probability), detail::gradient_edge(target)},
        [p, t, one = std::move(one), log_p = std::move(log_p), log_q = std::move(log_q)]
        (const auto& grad, auto& sink, const auto& edges) {
            if (sink.needs(edges[0])) {
                auto denominator = numeric::mul(p, numeric::sub(one, p));
                auto epsilon = numeric::full({}, 1e-12, p.dtype, p.device());
                denominator = numeric::where(numeric::compare(denominator, epsilon, numeric::Compare::lt),
                    epsilon, denominator);
                sink.add_owned(edges[0], numeric::mul(grad, numeric::div(numeric::sub(p, t), denominator)));
            }
            if (sink.needs(edges[1])) sink.add_owned(edges[1], numeric::mul(grad, numeric::sub(log_q, log_p)));
        });
    return reduce_loss(losses, reduction);
}

} // namespace dl
