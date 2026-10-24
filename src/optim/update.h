#pragma once

#include "dl/optim.h"
#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <type_traits>

namespace dl::optim::detail {

/// 状态按参数实际取得梯度的次数递推，独立调用与对象式批量调用共用这一准备过程。
dl::detail::numeric::AdamUpdate prepare_adam_update(Tensor& parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, const AdamWOptions& options, bool decoupled);

dl::detail::numeric::SGDUpdate prepare_sgd_update(Tensor& parameter, const Tensor& gradient,
    std::optional<Tensor>& velocity, const SGDOptions& options);

dl::detail::numeric::RMSpropUpdate prepare_rmsprop_update(Tensor& parameter, const Tensor& gradient,
    std::optional<RMSpropBuffers>& buffers, const RMSpropOptions& options);

dl::detail::numeric::AdagradUpdate prepare_adagrad_update(Tensor& parameter, const Tensor& gradient,
    std::optional<AdagradAccumulator>& accumulator, const AdagradOptions& options);

void step_adam_parameters(std::span<Tensor> parameters, std::span<const ParameterGroupState> groups,
    std::vector<std::optional<AdamMoments>>& states, const AdamWOptions& defaults, bool decoupled);

/// 梯度句柄覆盖描述符的借用期；分组配置逐项传入，后端负责设备和布局调度。
template <class State, class Options, class Prepare, class Submit>
void batch_parameters(std::span<Tensor> parameters, std::span<const ParameterGroupState> groups,
    std::vector<std::optional<State>>& states, const Options& defaults, Prepare prepare, Submit submit) {
    using Update = std::invoke_result_t<Prepare, Tensor&, const Tensor&, std::optional<State>&, const Options&>;
    std::vector<Tensor> gradients;
    std::vector<Update> updates;
    gradients.reserve(parameters.size());
    updates.reserve(parameters.size());
    for (const auto& group : groups) {
        auto options = defaults;
        options.learning_rate = group.learning_rate;
        options.weight_decay = group.weight_decay;
        for (const auto index : group.parameters) {
            auto gradient = parameters[index].grad();
            if (!gradient) continue;
            gradients.push_back(std::move(*gradient));
            updates.push_back(prepare(parameters[index], gradients.back(), states[index], options));
        }
    }
    submit(std::span<const Update>(updates));
}

} // namespace dl::optim::detail
