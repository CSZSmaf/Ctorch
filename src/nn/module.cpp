#include "dl/nn.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace dl::nn {
namespace {

std::vector<Tensor> values(StateDict state) {
    std::vector<Tensor> result;
    result.reserve(state.size());
    for (auto& [name, tensor] : state) result.push_back(std::move(tensor));
    return result;
}

} // namespace

Tensor Module::register_parameter(std::string name, Tensor value) {
    parameters_.emplace_back(std::move(name), value);
    return value;
}

Tensor Module::register_buffer(std::string name, Tensor value) {
    buffers_.emplace_back(std::move(name), value);
    return value;
}

void Module::register_module(std::string name, std::shared_ptr<Module> child) {
    children_.emplace_back(std::move(name), std::move(child));
}

StateDict Module::collect(bool buffers) const {
    StateDict result;
    std::unordered_set<const void*> seen;
    auto visit = [&](auto&& self, const Module& module, const std::string& prefix) -> void {
        const auto& local = buffers ? module.buffers_ : module.parameters_;
        for (const auto& [name, tensor] : local) {
            if (seen.insert(detail::TensorAccess::identity(tensor)).second) {
                result.emplace_back(prefix + name, tensor);
            }
        }
        for (const auto& [name, child] : module.children_) {
            self(self, *child, prefix + name + ".");
        }
    };
    visit(visit, *this, "");
    return result;
}

StateDict Module::named_parameters() const { return collect(false); }
StateDict Module::named_buffers() const { return collect(true); }

std::vector<Tensor> Module::parameters() const { return values(named_parameters()); }
std::vector<Tensor> Module::trainable_parameters() const { return values(named_trainable_parameters()); }

StateDict Module::named_trainable_parameters() const {
    auto result = named_parameters();
    std::erase_if(result, [](const NamedTensor& entry) { return !entry.second.requires_grad(); });
    return result;
}

void Module::requires_grad_(bool enabled) {
    for (const auto& [name, tensor] : named_parameters()) tensor.requires_grad_(enabled);
}

GradientModes Module::gradient_modes() const {
    GradientModes result;
    for (const auto& [name, tensor] : named_parameters()) result.emplace_back(name, tensor.requires_grad());
    return result;
}

void Module::load_gradient_modes(std::span<const std::pair<std::string, bool>> modes) {
    std::unordered_map<std::string_view, bool> source;
    for (const auto& [name, enabled] : modes) source.emplace(name, enabled);
    for (const auto& [name, tensor] : named_parameters()) tensor.requires_grad_(source.at(name));
}

StateDict Module::state_dict() const {
    auto result = named_parameters();
    for (auto& entry : named_buffers()) {
        result.push_back(std::move(entry));
    }
    for (auto& entry : result) {
        entry.second = entry.second.detach();
    }
    return result;
}

void Module::load_state_dict(std::span<const NamedTensor> state) {
    std::unordered_map<std::string_view, const Tensor*> source;
    for (const auto& [name, tensor] : state) {
        source.emplace(name, &tensor);
    }
    auto destination = state_dict();
    std::unordered_set<const detail::Storage*> target_storage;
    for (const auto& [name, tensor] : destination) {
        target_storage.insert(detail::data_of(tensor).storage.get());
    }

    // 状态表可借用当前参数或 buffer；先保存这些输入，使交换和视图加载使用调用时的值。
    std::vector<detail::TensorData> values;
    values.reserve(destination.size());
    for (const auto& [name, tensor] : destination) {
        const auto& value = detail::data_of(*source.at(name));
        values.push_back(target_storage.contains(value.storage.get()) ? detail::numeric::clone(value) : value);
    }
    for (std::size_t index = 0; index < destination.size(); ++index) {
        auto& tensor = destination[index].second;
        detail::numeric::copy_into(detail::TensorAccess::mutable_data(tensor), values[index]);
    }
}

void Module::train(bool enabled) {
    training_ = enabled;
    for (auto& entry : children_) {
        entry.second->train(enabled);
    }
}

TrainingModes Module::training_modes() const {
    TrainingModes result;
    auto visit = [&](auto&& self, const Module& module, const std::string& path) -> void {
        result.emplace_back(path, module.training_);
        for (const auto& [name, child] : module.children_) {
            self(self, *child, path.empty() ? name : path + "." + name);
        }
    };
    visit(visit, *this, "");
    return result;
}

void Module::load_training_modes(std::span<const std::pair<std::string, bool>> modes) {
    std::unordered_map<std::string_view, bool> source;
    for (const auto& [name, training] : modes) {
        source.emplace(name, training);
    }
    auto visit = [&](auto&& self, Module& module, const std::string& path) -> void {
        module.training_ = source.at(path);
        for (auto& [name, child] : module.children_) {
            self(self, *child, path.empty() ? name : path + "." + name);
        }
    };
    visit(visit, *this, "");
}

void Module::to(Device device) {
    auto tensors = named_parameters();
    for (auto& entry : named_buffers()) {
        tensors.push_back(std::move(entry));
    }
    for (auto& [name, tensor] : tensors) {
        auto& data = detail::TensorAccess::mutable_data(tensor);
        data = detail::numeric::to(data, device);
        auto* meta = detail::TensorAccess::autograd(tensor);
        if (meta && meta->leaf && meta->leaf->value) {
            meta->leaf->value = detail::numeric::to(*meta->leaf->value, device);
        }
    }
}

} // namespace dl::nn
