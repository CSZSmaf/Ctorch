#include "dl/amp.h"

#include "dl/ops.h"
#include "dl/optim.h"
#include "core/tensor_impl.h"
#include "ops/numeric.h"
#include "optim/gradients.h"

#include <algorithm>
#include <limits>

namespace dl {
namespace {

bool unscale_values(std::span<const Tensor> gradients, double inverse_scale) {
    struct Group { Device device; std::vector<detail::TensorData> gradients; };
    std::vector<Group> groups;
    for (const auto& gradient : gradients) {
        const auto device = gradient.device();
        auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& value) { return value.device == device; });
        if (group == groups.end()) {
            groups.push_back({device, {}});
            group = groups.end() - 1;
        }
        group->gradients.push_back(detail::data_of(gradient));
    }

    // 先提交所有设备的还原操作，再读取各自的一个汇总标志。
    std::vector<detail::TensorData> flags;
    flags.reserve(groups.size());
    for (auto& group : groups) {
        flags.push_back(detail::numeric::unscale_gradients(group.gradients, inverse_scale));
    }
    bool finite = true;
    for (const auto& flag : flags) {
        const auto host = detail::numeric::to(flag, Device::cpu());
        finite &= *host.data<float>() == 0.0f;
    }
    return finite;
}

} // namespace

GradScaler::GradScaler(GradScalerOptions options) : state_{options, static_cast<float>(options.initial_scale), 0} {}

Tensor GradScaler::scale(const Tensor& loss) const { return enabled() ? loss * state_.scale : loss; }

bool GradScaler::unscale(optim::Optimizer& optimizer) const {
    if (!enabled()) return true;
    return unscale_values(detail::leaf_gradients(optimizer.parameters()), 1.0 / state_.scale);
}

bool GradScaler::unscale_(std::span<const std::optional<Tensor>> gradients) const {
    if (!enabled()) return true;
    return unscale_values(detail::unique_gradients(gradients), 1.0 / state_.scale);
}

UnscaledGradients GradScaler::unscale(std::span<const std::optional<Tensor>> gradients) const {
    auto result = detail::clone_gradients(gradients);
    const auto finite = unscale_(result);
    return {std::move(result), finite};
}

void GradScaler::update(bool finite) {
    if (!enabled()) return;
    double factor = 1.0;
    if (!finite) {
        factor = state_.options.backoff_factor;
        state_.finite_steps = 0;
    } else if (++state_.finite_steps == state_.options.growth_interval) {
        factor = state_.options.growth_factor;
        state_.finite_steps = 0;
    }
    // 倍率和倒数保持可表示，连续回退仍允许倍率小于一。
    state_.scale = static_cast<float>(std::clamp(state_.scale * factor,
        static_cast<double>(std::numeric_limits<float>::min()),
        static_cast<double>(std::numeric_limits<float>::max())));
}

bool GradScaler::step(optim::Optimizer& optimizer) {
    const bool finite = unscale(optimizer);
    if (finite) optimizer.step();
    update(finite);
    return finite;
}

} // namespace dl
