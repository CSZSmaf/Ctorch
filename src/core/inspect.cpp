#include "dl/inspect.h"

#include <array>
#include <sstream>

namespace dl {
namespace {

void shape(std::ostream& out, const Shape& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? ", " : "") << values[i];
    out << ']';
}

void device(std::ostream& out, Device value) {
    if (value.type == DeviceType::cpu) out << "cpu";
    else out << "cuda:" << value.index;
}

} // namespace

std::string describe(const Tensor& tensor) {
    constexpr std::array names{"f32", "f64", "i64", "bool", "f16", "bf16"};
    std::ostringstream out;
    out << "Tensor(shape=";
    shape(out, tensor.shape());
    out << ", dtype=" << names[static_cast<std::size_t>(tensor.dtype())] << ", device=";
    device(out, tensor.device());
    out << ", strides=";
    shape(out, tensor.strides());
    out << ", offset=" << tensor.storage_offset()
        << ", requires_grad=" << std::boolalpha << tensor.requires_grad() << ')';
    return out.str();
}

std::string summary(const nn::Module& model) {
    std::ostringstream out;
    for (const auto& [path, training] : model.training_modes()) {
        out << (path.empty() ? "Module" : "  " + path) << (training ? " (train)\n" : " (eval)\n");
    }
    Index total = 0, trainable = 0;
    for (const auto& [name, value] : model.named_parameters()) {
        total += value.numel();
        if (value.requires_grad()) trainable += value.numel();
        out << "  " << name << ' ';
        shape(out, value.shape());
        out << " : " << value.numel() << (value.requires_grad() ? " trainable\n" : " frozen\n");
    }
    for (const auto& [alias, canonical] : model.state_aliases()) out << "  " << alias << " -> " << canonical << '\n';
    out << "Parameters: " << total << " unique, " << trainable << " trainable, " << total - trainable << " frozen";
    return out.str();
}

std::string describe(const plan::ExecutionPlan& plan) {
    std::ostringstream out;
    out << "ExecutionPlan(steps=" << plan.step_count() << ", fusions=" << plan.fusion_count()
        << ", slots=" << plan.slot_count() << ", inputs=" << plan.input_specs().size()
        << ", outputs=" << plan.output_specs().size() << ')';
    for (const auto& buffer : plan.buffer_requirements()) {
        out << "\n  ";
        device(out, buffer.device);
        out << " : " << buffer.count << " x " << buffer.bytes << " bytes";
    }
    return out.str();
}

} // namespace dl
