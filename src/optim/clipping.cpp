#include "dl/optim.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"
#include "optim/gradients.h"

namespace dl::optim {
namespace {

namespace numeric = dl::detail::numeric;

/// 数据描述持有相同存储，数值更新沿原布局写回，不创建求导节点或整张量副本。
std::vector<dl::detail::TensorData> gradient_data(std::span<const Tensor> values) {
    std::vector<dl::detail::TensorData> data;
    data.reserve(values.size());
    for (const auto& value : values) data.push_back(dl::detail::data_of(value));
    return data;
}

Tensor clip_norm(std::vector<Tensor> values, double max_norm, double order, TensorOptions empty = {}) {
    if (values.empty()) {
        return dl::detail::make_tensor(numeric::full({}, 0.0, empty.dtype, empty.device));
    }
    auto data = gradient_data(values);
    return dl::detail::make_tensor(numeric::clip_grad_norm(data, max_norm, order));
}

void clip_value(std::vector<Tensor> values, double value) {
    auto data = gradient_data(values);
    numeric::clip_grad_value(data, value);
}

} // namespace

Tensor clip_grad_norm(std::span<const Tensor> parameters, double max_norm, double order) {
    const auto empty = parameters.empty() ? TensorOptions{}
        : TensorOptions{.dtype = parameters.front().dtype(), .device = parameters.front().device()};
    return clip_norm(dl::detail::leaf_gradients(parameters), max_norm, order, empty);
}

void clip_grad_value(std::span<const Tensor> parameters, double value) {
    clip_value(dl::detail::leaf_gradients(parameters), value);
}

Tensor clip_gradients_norm_(std::span<const std::optional<Tensor>> gradients, double max_norm, double order) {
    return clip_norm(dl::detail::unique_gradients(gradients), max_norm, order);
}

ClipGradNormResult clip_gradients_norm(std::span<const std::optional<Tensor>> gradients,
                                     double max_norm, double order) {
    auto result = dl::detail::clone_gradients(gradients);
    auto norm = clip_gradients_norm_(result, max_norm, order);
    return {std::move(result), std::move(norm)};
}

void clip_gradients_value_(std::span<const std::optional<Tensor>> gradients, double value) {
    clip_value(dl::detail::unique_gradients(gradients), value);
}

Gradients clip_gradients_value(std::span<const std::optional<Tensor>> gradients, double value) {
    auto result = dl::detail::clone_gradients(gradients);
    clip_gradients_value_(result, value);
    return result;
}

} // namespace dl::optim
