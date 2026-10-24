#pragma once

#include "dl/amp.h"

namespace dl::detail {

/// 按值采集当前线程的两类设备精度，供反向重算和固定计划沿用前向配置。
struct AutocastState {
    struct Precision { DType dtype; bool enabled; };
    Precision cpu{autocast_dtype(DeviceType::cpu), is_autocast_enabled(DeviceType::cpu)};
    Precision cuda{autocast_dtype(DeviceType::cuda), is_autocast_enabled(DeviceType::cuda)};
};

/// 在当前调用内采用保存的精度；两个普通 guard 在离开时恢复各自的外层状态。
class AutocastReplay {
public:
    explicit AutocastReplay(const AutocastState& state)
        : cpu_(DeviceType::cpu, state.cpu.dtype, state.cpu.enabled),
          cuda_(DeviceType::cuda, state.cuda.dtype, state.cuda.enabled) {}

private:
    AutocastGuard cpu_, cuda_;
};

} // namespace dl::detail
