#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_status dl_cumsum(const dl_tensor* input, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] { return dl::cumsum(input->value, axis); });
}
dl_status dl_argsort(const dl_tensor* input, int64_t axis, int32_t descending, dl_tensor** output) {
    return make_handle(output, [&] { return dl::argsort(input->value, axis, descending != 0); });
}
dl_status dl_sort(const dl_tensor* input, int64_t axis, int32_t descending, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto result = dl::sort(input->value, axis, descending != 0);
        return std::vector<dl::Tensor>{std::move(result.values), std::move(result.indices)};
    });
}
dl_status dl_topk(const dl_tensor* input, int64_t k, int64_t axis, int32_t largest, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto result = dl::topk(input->value, k, axis, largest != 0);
        return std::vector<dl::Tensor>{std::move(result.values), std::move(result.indices)};
    });
}
dl_status dl_pad(const dl_tensor* input, dl_shape padding, double value, dl_tensor** output) {
    return make_handle(output, [&] { return dl::pad(input->value, convert(padding), value); });
}
dl_status dl_one_hot(const dl_tensor* input, int64_t classes, dl_dtype dtype, dl_tensor** output) {
    return make_handle(output, [&] { return dl::one_hot(input->value, classes, convert(dtype)); });
}
dl_status dl_cross_entropy_weighted(const dl_tensor* logits, const dl_tensor* labels,
    const dl_tensor* weight, double smoothing, const int64_t* ignore, dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] {
        dl::CrossEntropyOptions options;
        options.reduction = convert(reduction);
        if (ignore) options.ignore_index = *ignore;
        if (weight) options.weight = weight->value;
        options.label_smoothing = smoothing;
        return dl::cross_entropy(logits->value, labels->value, std::move(options));
    });
}
dl_status dl_binary_cross_entropy_weighted(const dl_tensor* logits, const dl_tensor* target,
    const dl_tensor* weight, const dl_tensor* positive_weight, dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] {
        dl::BinaryLossOptions options;
        options.reduction = convert(reduction);
        if (weight) options.weight = weight->value;
        if (positive_weight) options.positive_weight = positive_weight->value;
        return dl::binary_cross_entropy_with_logits(logits->value, target->value, std::move(options));
    });
}
dl_status dl_focal_loss(const dl_tensor* logits, const dl_tensor* target, double gamma, double alpha,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::focal_loss(logits->value, target->value, gamma, alpha, convert(reduction)); });
}
dl_status dl_dice_loss(const dl_tensor* probability, const dl_tensor* target, dl_shape axes, double epsilon,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::dice_loss(probability->value, target->value, convert(axes), epsilon, convert(reduction)); });
}
dl_status dl_kl_div(const dl_tensor* input, const dl_tensor* target, dl_loss_reduction reduction,
    int32_t log_target, dl_tensor** output) {
    return make_handle(output, [&] { return dl::kl_div(input->value, target->value, convert(reduction), log_target != 0); });
}

} // extern "C"
