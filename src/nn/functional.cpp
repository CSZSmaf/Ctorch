#include "dl/functional.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <unordered_map>

namespace dl::nn {
namespace {

// 绑定只覆盖同步前向的当前线程；嵌套调用保存并恢复外层上下文。
thread_local functional_detail::FunctionalContext* current = nullptr;

Tensor copy_value(const Tensor& value, bool requires_grad = false) {
    return dl::detail::make_tensor(dl::detail::numeric::clone(dl::detail::data_of(value)), requires_grad);
}

} // namespace

namespace functional_detail {

struct FunctionalLayout {
    std::vector<Tensor> parameters, buffers;
    std::vector<std::string> parameter_names, buffer_names, module_names;
    std::vector<const Module*> modules;
    std::unordered_map<const void*, std::size_t> parameter_slots, buffer_slots;
    std::unordered_map<const Module*, std::size_t> module_slots;
};

struct FunctionalContext {
    const FunctionalLayout& layout;
    std::span<const Tensor> parameters;
    std::vector<Tensor> buffers;
    std::span<const std::uint8_t> training;
    RandomKey key;
    std::vector<std::uint64_t> random_calls;
    FunctionalContext* previous;

    FunctionalContext(const FunctionalLayout& layout, std::span<const Tensor> parameters,
                      std::span<const Tensor> initial_buffers, std::span<const std::uint8_t> training, RandomKey key)
        : layout(layout), parameters(parameters), training(training), key(key),
          random_calls(layout.modules.size()), previous(current) {
        buffers.reserve(initial_buffers.size());
        for (const auto& buffer : initial_buffers) buffers.push_back(copy_value(buffer));
        current = this;
    }

    ~FunctionalContext() { current = previous; }
};

} // namespace functional_detail

namespace {

struct ModuleBinding {
    functional_detail::FunctionalContext* context = nullptr;
    std::size_t slot = 0;
};

// 从内到外选择实际包含本模块的绑定；独立模块继续读取自己的参数、模式与随机流。
ModuleBinding binding_for(const Module* module) {
    for (auto* context = current; context; context = context->previous) {
        const auto position = context->layout.module_slots.find(module);
        if (position != context->layout.module_slots.end()) return {context, position->second};
    }
    return {};
}

} // namespace

FunctionalModule::FunctionalModule(Module& module) {
    auto layout = std::make_shared<functional_detail::FunctionalLayout>();
    auto collect = [](StateDict entries, auto& values, auto& names, auto& slots) {
        for (auto& [name, tensor] : entries) {
            slots.emplace(dl::detail::TensorAccess::identity(tensor), values.size());
            names.push_back(std::move(name));
            values.push_back(std::move(tensor));
        }
    };
    collect(module.named_parameters(), layout->parameters, layout->parameter_names, layout->parameter_slots);
    collect(module.named_buffers(), layout->buffers, layout->buffer_names, layout->buffer_slots);

    auto visit = [&](auto&& self, const Module& node, const std::string& name) -> void {
        if (!layout->module_slots.emplace(&node, layout->modules.size()).second) return;
        layout->modules.push_back(&node);
        layout->module_names.push_back(name);
        for (const auto& [child_name, child] : node.children_) {
            self(self, *child, name.empty() ? child_name : name + "." + child_name);
        }
    };
    visit(visit, module, "");
    layout_ = std::move(layout);
}

std::span<const std::string> FunctionalModule::parameter_names() const { return layout_->parameter_names; }
std::span<const std::string> FunctionalModule::buffer_names() const { return layout_->buffer_names; }
std::span<const std::string> FunctionalModule::module_names() const { return layout_->module_names; }

FunctionalState FunctionalModule::state() const {
    FunctionalState result;
    result.parameters.reserve(layout_->parameters.size());
    result.buffers.reserve(layout_->buffers.size());
    result.training.reserve(layout_->modules.size());
    for (const auto& parameter : layout_->parameters) {
        result.parameters.push_back(copy_value(parameter, parameter.requires_grad()));
    }
    for (const auto& buffer : layout_->buffers) result.buffers.push_back(copy_value(buffer));
    for (const auto* module : layout_->modules) result.training.push_back(module->training_);
    return result;
}

FunctionalModule::Invocation::Invocation(const FunctionalModule& module, std::span<const Tensor> parameters,
    std::span<const Tensor> buffers, std::span<const std::uint8_t> training, RandomKey key)
    : context_(std::make_unique<functional_detail::FunctionalContext>(
          *module.layout_, parameters, buffers, training, key)) {}

FunctionalModule::Invocation::~Invocation() = default;
std::vector<Tensor> FunctionalModule::Invocation::take_buffers() { return std::move(context_->buffers); }

const Tensor& Module::bound(const Tensor& value) const {
    const auto binding = binding_for(this);
    if (!binding.context) return value;
    const auto& context = *binding.context;
    return context.parameters[context.layout.parameter_slots.at(dl::detail::TensorAccess::identity(value))];
}

std::optional<Tensor> Module::bound(const std::optional<Tensor>& value) const {
    return value ? std::optional{bound(*value)} : std::nullopt;
}

const Tensor& Module::bound_buffer(const Tensor& value) const {
    const auto binding = binding_for(this);
    if (!binding.context) return value;
    const auto& context = *binding.context;
    return context.buffers[context.layout.buffer_slots.at(dl::detail::TensorAccess::identity(value))];
}

bool Module::is_training() const {
    const auto binding = binding_for(this);
    return binding.context ? binding.context->training[binding.slot] != 0 : training_;
}

std::optional<RandomKey> Module::next_random_key() const {
    const auto binding = binding_for(this);
    if (!binding.context) return std::nullopt;
    auto& context = *binding.context;
    return fold_in(fold_in(context.key, static_cast<std::uint64_t>(binding.slot)), context.random_calls[binding.slot]++);
}

Tensor Module::apply_dropout(const Tensor& input, double probability) const {
    if (auto key = next_random_key()) return dropout(input, probability, *key, is_training());
    return dropout(input, probability, is_training());
}

} // namespace dl::nn
