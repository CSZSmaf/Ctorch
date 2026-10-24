#include "dl/scheduler.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>

namespace dl::optim {
namespace {

double rate(const Schedule& schedule, Index step, double base) {
    return std::visit([&](const auto& policy) {
        using Policy = std::decay_t<decltype(policy)>;
        if constexpr (std::is_same_v<Policy, StepSchedule>) {
            return base * std::pow(policy.gamma, step / policy.interval);
        } else if constexpr (std::is_same_v<Policy, MultiStepSchedule>) {
            const auto count = std::upper_bound(policy.milestones.begin(), policy.milestones.end(), step)
                - policy.milestones.begin();
            return base * std::pow(policy.gamma, count);
        } else if constexpr (std::is_same_v<Policy, ExponentialSchedule>) {
            return base * std::pow(policy.gamma, step);
        } else if constexpr (std::is_same_v<Policy, CosineSchedule>) {
            return policy.minimum_rate + (base - policy.minimum_rate)
                * (1.0 + std::cos(std::numbers::pi * static_cast<double>(step) / policy.period)) / 2.0;
        } else {
            const double progress = static_cast<double>(std::min(step, policy.steps)) / policy.steps;
            return base * (policy.start_factor + (policy.end_factor - policy.start_factor) * progress);
        }
    }, schedule);
}

} // namespace

LRScheduler::LRScheduler(Optimizer& optimizer, Schedule schedule)
    : optimizer_(optimizer), state_{std::move(schedule), 0, {}} {
    for (const auto& group : optimizer.groups()) state_.base_rates.push_back(group.learning_rate);
    apply();
}

void LRScheduler::apply() {
    for (std::size_t group = 0; group < state_.base_rates.size(); ++group) {
        optimizer_.set_learning_rate(rate(state_.schedule, state_.step, state_.base_rates[group]), group);
    }
}

void LRScheduler::step() { step(state_.step + 1); }

void LRScheduler::step(Index completed_steps) {
    state_.step = completed_steps;
    apply();
}

void LRScheduler::load_state_dict(const SchedulerState& state) {
    state_ = state;
    apply();
}

} // namespace dl::optim
