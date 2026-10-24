#pragma once

#include "dl/optim.h"

namespace dl::optim {

struct StepSchedule { Index interval; double gamma = 0.1; };
struct MultiStepSchedule { Shape milestones; double gamma = 0.1; };
struct ExponentialSchedule { double gamma; };
struct CosineSchedule { Index period; double minimum_rate = 0.0; };
struct LinearSchedule { Index steps; double start_factor = 1.0 / 3.0; double end_factor = 1.0; };
using Schedule = std::variant<StepSchedule, MultiStepSchedule, ExponentialSchedule, CosineSchedule, LinearSchedule>;

struct SchedulerState {
    Schedule schedule;
    Index step = 0;
    std::vector<double> base_rates;
};

/// 调度器借用优化器，构造时应用第零步，在 optimizer.step() 后推进一次。
/// interval、period、steps 为正；milestones 按非递减顺序给出，重复里程碑重复衰减。
/// 每组学习率由创建时的基准值与当前步数直接计算，加载恢复公式、基准值和步数。
class DL_API LRScheduler {
public:
    LRScheduler(Optimizer& optimizer, Schedule schedule);
    virtual ~LRScheduler() = default;
    void step();
    void step(Index completed_steps);
    Index step_count() const { return state_.step; }
    SchedulerState state_dict() const { return state_; }
    void load_state_dict(const SchedulerState& state);

private:
    void apply();
    Optimizer& optimizer_;
    SchedulerState state_;
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, Index interval, double gamma = 0.1)
        : LRScheduler(optimizer, StepSchedule{interval, gamma}) {}
};

class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& optimizer, Shape milestones, double gamma = 0.1)
        : LRScheduler(optimizer, MultiStepSchedule{std::move(milestones), gamma}) {}
};

class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& optimizer, double gamma) : LRScheduler(optimizer, ExponentialSchedule{gamma}) {}
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, Index period, double minimum_rate = 0.0)
        : LRScheduler(optimizer, CosineSchedule{period, minimum_rate}) {}
};

class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer, Index steps, double start_factor = 1.0 / 3.0, double end_factor = 1.0)
        : LRScheduler(optimizer, LinearSchedule{steps, start_factor, end_factor}) {}
};

} // namespace dl::optim
