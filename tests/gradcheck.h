#pragma once

#include "test_support.h"

#include <functional>

namespace test {

inline void directional_check(std::vector<double> values, dl::Shape shape,
                               const std::function<dl::Tensor(const dl::Tensor&)>& function) {
    const double step = 1e-6;
    auto input = dl::from_span<double>(values, shape, true);
    function(input).backward();
    auto gradient = input.grad()->to_vector<double>();
    auto plus = values;
    auto minus = values;
    double projection = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double direction = 0.2 + 0.1 * static_cast<double>(index % 3);
        plus[index] += step * direction;
        minus[index] -= step * direction;
        projection += direction * gradient[index];
    }
    dl::NoGradGuard guard;
    const auto upper = function(dl::from_span<double>(plus, shape)).item<double>();
    const auto lower = function(dl::from_span<double>(minus, shape)).item<double>();
    close(projection, (upper - lower) / (2 * step), 1e-7, 1e-5);
}

} // namespace test
