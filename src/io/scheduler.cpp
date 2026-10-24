#include "io/training_archive.h"

#include <stdexcept>
#include <type_traits>

namespace dl::io::archive {

void write_scheduler(std::ostream& output, const optim::SchedulerState& state) {
    constexpr std::array names{"step", "multistep", "exponential", "cosine", "linear"};
    write_text(output, names[state.schedule.index()]);
    auto integer = [&](Index value) { write_integer(output, static_cast<std::uint64_t>(value)); };
    std::visit([&](const auto& policy) {
        using Policy = std::decay_t<decltype(policy)>;
        if constexpr (std::is_same_v<Policy, optim::StepSchedule>) {
            integer(policy.interval);
            write_double(output, policy.gamma);
        } else if constexpr (std::is_same_v<Policy, optim::MultiStepSchedule>) {
            integer(static_cast<Index>(policy.milestones.size()));
            for (auto milestone : policy.milestones) integer(milestone);
            write_double(output, policy.gamma);
        } else if constexpr (std::is_same_v<Policy, optim::ExponentialSchedule>) {
            write_double(output, policy.gamma);
        } else if constexpr (std::is_same_v<Policy, optim::CosineSchedule>) {
            integer(policy.period);
            write_double(output, policy.minimum_rate);
        } else {
            integer(policy.steps);
            write_double(output, policy.start_factor);
            write_double(output, policy.end_factor);
        }
    }, state.schedule);
    integer(state.step);
    integer(static_cast<Index>(state.base_rates.size()));
    for (double value : state.base_rates) write_double(output, value);
}

optim::SchedulerState read_scheduler(std::istream& input) {
    auto integer = [&] { return static_cast<Index>(read_integer<std::uint64_t>(input)); };
    const auto kind = read_text(input);
    optim::SchedulerState result;
    if (kind == "step") result.schedule = optim::StepSchedule{integer(), read_double(input)};
    else if (kind == "multistep") {
        Shape milestones(static_cast<std::size_t>(integer()));
        for (auto& milestone : milestones) milestone = integer();
        result.schedule = optim::MultiStepSchedule{std::move(milestones), read_double(input)};
    } else if (kind == "exponential") result.schedule = optim::ExponentialSchedule{read_double(input)};
    else if (kind == "cosine") result.schedule = optim::CosineSchedule{integer(), read_double(input)};
    else if (kind == "linear") result.schedule = optim::LinearSchedule{integer(), read_double(input), read_double(input)};
    else throw std::runtime_error("unsupported checkpoint scheduler: " + kind);
    result.step = integer();
    result.base_rates.resize(static_cast<std::size_t>(integer()));
    for (auto& value : result.base_rates) value = read_double(input);
    return result;
}

} // namespace dl::io::archive
