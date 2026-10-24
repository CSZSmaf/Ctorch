#pragma once

#include "dl/optim.h"

namespace dl::optim::detail {

inline Tensor step_tensor(Index step) {
    return from_span<Index>(std::span<const Index>(&step, 1), {});
}

/// C17 和固定计划共用结果顺序；计数作为 CPU int64 标量交接。
inline std::vector<Tensor> result_tensors(SGDResult result) {
    std::vector<Tensor> values{std::move(result.parameter)};
    if (result.velocity) values.push_back(std::move(*result.velocity));
    return values;
}

inline std::vector<Tensor> result_tensors(AdamWResult result) {
    return {std::move(result.parameter), std::move(result.moments.first),
        std::move(result.moments.second), step_tensor(result.moments.step)};
}

inline std::vector<Tensor> result_tensors(RMSpropResult result) {
    std::vector<Tensor> values{std::move(result.parameter), std::move(result.buffers.square_average)};
    if (result.buffers.momentum) values.push_back(std::move(*result.buffers.momentum));
    if (result.buffers.gradient_average) values.push_back(std::move(*result.buffers.gradient_average));
    values.push_back(step_tensor(result.buffers.step));
    return values;
}

inline std::vector<Tensor> result_tensors(AdagradResult result) {
    return {std::move(result.parameter), std::move(result.accumulator.sum), step_tensor(result.accumulator.step)};
}

} // namespace dl::optim::detail
