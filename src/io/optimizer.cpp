#include "io/training_archive.h"

#include <stdexcept>
#include <type_traits>

namespace dl::io::archive {
namespace {

void write_numbers(std::ostream& output, std::initializer_list<double> values) {
    for (double value : values) write_double(output, value);
}

void write_options(std::ostream& out, const optim::SGDOptions& v) {
    write_numbers(out, {v.learning_rate, v.momentum, v.dampening, v.weight_decay});
    write_integer(out, static_cast<std::uint8_t>(v.nesterov));
}
void write_options(std::ostream& out, const optim::AdamWOptions& v) {
    write_numbers(out, {v.learning_rate, v.beta1, v.beta2, v.epsilon, v.weight_decay});
}
void write_options(std::ostream& out, const optim::AdamOptions& v) {
    write_numbers(out, {v.learning_rate, v.beta1, v.beta2, v.epsilon, v.weight_decay});
}
void write_options(std::ostream& out, const optim::RMSpropOptions& v) {
    write_numbers(out, {v.learning_rate, v.alpha, v.epsilon, v.weight_decay, v.momentum});
    write_integer(out, static_cast<std::uint8_t>(v.centered));
}
void write_options(std::ostream& out, const optim::AdagradOptions& v) {
    write_numbers(out, {v.learning_rate, v.learning_rate_decay, v.weight_decay, v.initial_accumulator, v.epsilon});
}

template <class Options> Options read_options(std::istream& in) {
    if constexpr (std::is_same_v<Options, optim::SGDOptions>) {
        return {read_double(in), read_double(in), read_double(in), read_double(in), read_integer<std::uint8_t>(in) != 0};
    } else if constexpr (std::is_same_v<Options, optim::RMSpropOptions>) {
        return {read_double(in), read_double(in), read_double(in), read_double(in), read_double(in), read_integer<std::uint8_t>(in) != 0};
    } else {
        return {read_double(in), read_double(in), read_double(in), read_double(in), read_double(in)};
    }
}

void write_optional(std::ostream& output, const std::optional<Tensor>& value) {
    write_integer(output, static_cast<std::uint8_t>(value.has_value()));
    if (value) write_tensor(output, *value);
}
std::optional<Tensor> read_optional(std::istream& input) {
    return read_integer<std::uint8_t>(input) != 0 ? std::optional{read_tensor(input)} : std::nullopt;
}

void write_slot(std::ostream& output, const Tensor& value) { write_tensor(output, value); }
void write_slot(std::ostream& output, const optim::AdamMoments& value) {
    write_tensor(output, value.first);
    write_tensor(output, value.second);
    write_integer(output, static_cast<std::uint64_t>(value.step));
}
void write_slot(std::ostream& output, const optim::RMSpropBuffers& value) {
    write_tensor(output, value.square_average);
    write_optional(output, value.momentum);
    write_optional(output, value.gradient_average);
    write_integer(output, static_cast<std::uint64_t>(value.step));
}
void write_slot(std::ostream& output, const optim::AdagradAccumulator& value) {
    write_tensor(output, value.sum);
    write_integer(output, static_cast<std::uint64_t>(value.step));
}

template <class Slot> Slot read_slot(std::istream& in) {
    if constexpr (std::is_same_v<Slot, Tensor>) return read_tensor(in);
    else if constexpr (std::is_same_v<Slot, optim::AdamMoments>) {
        return {read_tensor(in), read_tensor(in), static_cast<Index>(read_integer<std::uint64_t>(in))};
    } else if constexpr (std::is_same_v<Slot, optim::RMSpropBuffers>) {
        return {read_tensor(in), read_optional(in), read_optional(in), static_cast<Index>(read_integer<std::uint64_t>(in))};
    } else return {read_tensor(in), static_cast<Index>(read_integer<std::uint64_t>(in))};
}

void write_groups(std::ostream& output, const std::vector<optim::ParameterGroupState>& groups) {
    write_integer(output, static_cast<std::uint64_t>(groups.size()));
    for (const auto& group : groups) {
        write_integer(output, static_cast<std::uint64_t>(group.parameters.size()));
        for (const auto index : group.parameters) write_integer(output, static_cast<std::uint64_t>(index));
        write_numbers(output, {group.learning_rate, group.weight_decay});
    }
}

std::vector<optim::ParameterGroupState> read_groups(std::istream& input) {
    std::vector<optim::ParameterGroupState> result(static_cast<std::size_t>(read_integer<std::uint64_t>(input)));
    for (auto& group : result) {
        group.parameters.resize(static_cast<std::size_t>(read_integer<std::uint64_t>(input)));
        for (auto& index : group.parameters) index = static_cast<std::size_t>(read_integer<std::uint64_t>(input));
        group.learning_rate = read_double(input);
        group.weight_decay = read_double(input);
    }
    return result;
}

template <class State> State read_state(std::istream& input, bool with_groups) {
    State state;
    state.options = read_options<decltype(state.options)>(input);
    auto& slots = parameter_slots(state);
    using Slot = typename std::decay_t<decltype(slots)>::value_type::value_type;
    slots.resize(static_cast<std::size_t>(read_integer<std::uint64_t>(input)));
    for (auto& slot : slots) {
        if (read_integer<std::uint8_t>(input) != 0) slot = read_slot<Slot>(input);
    }
    if (with_groups) state.groups = read_groups(input);
    return state;
}

} // namespace

void write_optimizer(std::ostream& output, const optim::OptimizerState& state) {
    constexpr std::array names{"SGD", "AdamW", "Adam", "RMSprop", "Adagrad"};
    write_text(output, names[state.index()]);
    std::visit([&](const auto& value) {
        write_options(output, value.options);
        const auto& slots = parameter_slots(value);
        write_integer(output, static_cast<std::uint64_t>(slots.size()));
        for (const auto& slot : slots) {
            write_integer(output, static_cast<std::uint8_t>(slot.has_value()));
            if (slot) write_slot(output, *slot);
        }
        write_groups(output, value.groups);
    }, state);
}

optim::OptimizerState read_optimizer(std::istream& input, bool with_groups) {
    const auto algorithm = read_text(input);
    if (algorithm == "SGD") return read_state<optim::SGDState>(input, with_groups);
    if (algorithm == "AdamW") return read_state<optim::AdamWState>(input, with_groups);
    if (algorithm == "Adam") return read_state<optim::AdamState>(input, with_groups);
    if (algorithm == "RMSprop") return read_state<optim::RMSpropState>(input, with_groups);
    if (algorithm == "Adagrad") return read_state<optim::AdagradState>(input, with_groups);
    throw std::runtime_error("unsupported checkpoint optimizer: " + algorithm);
}

} // namespace dl::io::archive
