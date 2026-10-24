#pragma once

#include "dl/optim.h"

namespace dl::optim::detail {

/// 各算法按同一参数槽顺序保存状态，具体缓冲结构继续由算法本身表达。
template <class State> auto& parameter_slots(State& state) {
    if constexpr (requires { state.velocity; }) return state.velocity;
    else if constexpr (requires { state.moments; }) return state.moments;
    else if constexpr (requires { state.buffers; }) return state.buffers;
    else return state.accumulators;
}

/// 完整排列的第 i 项指明新槽 i 对应的旧槽，参数组索引按同一映射更新。
/// 操作移动状态句柄，保留算法配置、每参数计数与组顺序，张量数值不在此复制。
OptimizerState reorder_state(OptimizerState state, std::span<const std::size_t> source_for_target);

} // namespace dl::optim::detail
