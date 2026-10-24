#pragma once

#include "optim/state.h"
#include "dl/scheduler.h"
#include "io/archive.h"

namespace dl::io::archive {

using optim::detail::parameter_slots;

Tensor copy_tensor(const Tensor& value, Device device);
optim::OptimizerState copy_optimizer(optim::OptimizerState state, Device device);
optim::OptimizerState reorder_optimizer(optim::OptimizerState state,
    std::span<const std::string> source_names, std::span<const std::string> target_names);

void write_optimizer(std::ostream& output, const optim::OptimizerState& state);
optim::OptimizerState read_optimizer(std::istream& input, bool with_groups);
void write_scheduler(std::ostream& output, const optim::SchedulerState& state);
optim::SchedulerState read_scheduler(std::istream& input);

} // namespace dl::io::archive
