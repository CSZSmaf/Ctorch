#include "optim/state.h"

#include <type_traits>

namespace dl::optim::detail {

OptimizerState reorder_state(OptimizerState state, std::span<const std::size_t> source_for_target) {
    std::vector<std::size_t> target_for_source(source_for_target.size());
    for (std::size_t i = 0; i < source_for_target.size(); ++i) target_for_source[source_for_target[i]] = i;
    std::visit([&](auto& value) {
        auto& slots = parameter_slots(value);
        auto previous = std::move(slots);
        slots.resize(source_for_target.size());
        for (std::size_t i = 0; i < source_for_target.size(); ++i) {
            slots[i] = std::move(previous[source_for_target[i]]);
        }
        for (auto& group : value.groups) {
            for (auto& index : group.parameters) index = target_for_source[index];
        }
    }, state);
    return state;
}

} // namespace dl::optim::detail

namespace dl::optim {
namespace {

Tensor copy_buffer(const Tensor& tensor, Device device) {
    // 同设备也复制逻辑元素，转置、切片与广播状态由此成为独立连续缓冲。
    auto value = tensor.detach();
    return value.device() == device ? value.clone() : value.to(device);
}

std::optional<Tensor> copy_buffer(const std::optional<Tensor>& tensor, Device device) {
    return tensor ? std::optional{copy_buffer(*tensor, device)} : std::nullopt;
}

AdamMoments copy_buffer(const AdamMoments& value, Device device) {
    return {copy_buffer(value.first, device), copy_buffer(value.second, device), value.step};
}

RMSpropBuffers copy_buffer(const RMSpropBuffers& value, Device device) {
    return {copy_buffer(value.square_average, device), copy_buffer(value.momentum, device),
        copy_buffer(value.gradient_average, device), value.step};
}

AdagradAccumulator copy_buffer(const AdagradAccumulator& value, Device device) {
    return {copy_buffer(value.sum, device), value.step};
}

template <class Slot>
void restore(std::vector<std::optional<Slot>>& target, const std::vector<std::optional<Slot>>& source,
    std::span<const Tensor> parameters) {
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        target[index] = source[index] ? std::optional{copy_buffer(*source[index], parameters[index].device())} : std::nullopt;
    }
}

} // namespace

std::unique_ptr<Optimizer> from_state(std::vector<Tensor> parameters, const OptimizerState& state) {
    auto result = std::visit([&](const auto& saved) -> std::unique_ptr<Optimizer> {
        using State = std::decay_t<decltype(saved)>;
        if constexpr (std::is_same_v<State, SGDState>) return std::make_unique<SGD>(std::move(parameters), saved.options);
        else if constexpr (std::is_same_v<State, AdamWState>) return std::make_unique<AdamW>(std::move(parameters), saved.options);
        else if constexpr (std::is_same_v<State, AdamState>) return std::make_unique<Adam>(std::move(parameters), saved.options);
        else if constexpr (std::is_same_v<State, RMSpropState>) return std::make_unique<RMSprop>(std::move(parameters), saved.options);
        else return std::make_unique<Adagrad>(std::move(parameters), saved.options);
    }, state);
    result->load_state_dict(state);
    return result;
}

OptimizerState SGD::state_dict() const { return SGDState{options_, velocity_, groups_}; }

void SGD::load_state_dict(const OptimizerState& state) {
    const auto& saved = std::get<SGDState>(state);
    options_ = saved.options;
    load_groups(saved.groups, options_.learning_rate, options_.weight_decay);
    restore(velocity_, saved.velocity, parameters_);
}

OptimizerState AdamW::state_dict() const { return AdamWState{options_, states_, groups_}; }

void AdamW::load_state_dict(const OptimizerState& state) {
    const auto& saved = std::get<AdamWState>(state);
    options_ = saved.options;
    load_groups(saved.groups, options_.learning_rate, options_.weight_decay);
    restore(states_, saved.moments, parameters_);
}

OptimizerState Adam::state_dict() const { return AdamState{options_, states_, groups_}; }

void Adam::load_state_dict(const OptimizerState& state) {
    const auto& saved = std::get<AdamState>(state);
    options_ = saved.options;
    load_groups(saved.groups, options_.learning_rate, options_.weight_decay);
    restore(states_, saved.moments, parameters_);
}

OptimizerState RMSprop::state_dict() const { return RMSpropState{options_, states_, groups_}; }

void RMSprop::load_state_dict(const OptimizerState& state) {
    const auto& saved = std::get<RMSpropState>(state);
    options_ = saved.options;
    load_groups(saved.groups, options_.learning_rate, options_.weight_decay);
    restore(states_, saved.buffers, parameters_);
}

OptimizerState Adagrad::state_dict() const { return AdagradState{options_, states_, groups_}; }

void Adagrad::load_state_dict(const OptimizerState& state) {
    const auto& saved = std::get<AdagradState>(state);
    options_ = saved.options;
    load_groups(saved.groups, options_.learning_rate, options_.weight_decay);
    restore(states_, saved.accumulators, parameters_);
}

} // namespace dl::optim
