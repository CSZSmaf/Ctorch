#include "io/training_archive.h"

#include <string_view>
#include <unordered_map>

namespace dl::io::archive {
namespace {

void copy_buffer(Tensor& value, Device device) { value = copy_tensor(value, device); }
void copy_buffer(optim::AdamMoments& value, Device device) {
    copy_buffer(value.first, device);
    copy_buffer(value.second, device);
}
void copy_buffer(optim::RMSpropBuffers& value, Device device) {
    copy_buffer(value.square_average, device);
    if (value.momentum) copy_buffer(*value.momentum, device);
    if (value.gradient_average) copy_buffer(*value.gradient_average, device);
}
void copy_buffer(optim::AdagradAccumulator& value, Device device) { copy_buffer(value.sum, device); }

} // namespace

Tensor copy_tensor(const Tensor& value, Device device) {
    auto detached = value.detach();
    return detached.device() == device ? detached.clone() : detached.to(device);
}

optim::OptimizerState copy_optimizer(optim::OptimizerState state, Device device) {
    std::visit([&](auto& value) {
        for (auto& slot : parameter_slots(value)) if (slot) copy_buffer(*slot, device);
    }, state);
    return state;
}

optim::OptimizerState reorder_optimizer(optim::OptimizerState state,
    std::span<const std::string> source_names, std::span<const std::string> target_names) {
    std::unordered_map<std::string_view, std::size_t> source;
    for (std::size_t i = 0; i < source_names.size(); ++i) source.emplace(source_names[i], i);
    std::vector<std::size_t> order;
    order.reserve(target_names.size());
    for (const auto& name : target_names) order.push_back(source.at(name));
    return optim::detail::reorder_state(std::move(state), order);
}

} // namespace dl::io::archive
