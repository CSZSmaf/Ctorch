#pragma once

#include <dl/dl.h>

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace test {

inline dl::Tensor tensor(std::initializer_list<double> values, dl::Shape shape, bool grad = false) {
    return dl::from_span<double>(std::span(values.begin(), values.size()), std::move(shape), grad);
}

inline dl::Tensor indices(std::initializer_list<dl::Index> values) {
    return dl::from_span<dl::Index>(std::span(values.begin(), values.size()),
                                   {static_cast<dl::Index>(values.size())});
}

inline void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void close(double actual, double expected, double absolute = 1e-10, double relative = 1e-9) {
    if (!(std::abs(actual - expected) <= absolute + relative * std::abs(expected))) {
        throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

inline void values(const dl::Tensor& actual, std::initializer_list<double> expected,
                   double absolute = 1e-10, double relative = 1e-9) {
    const auto items = actual.astype(dl::DType::f64).to_vector<double>();
    expect(items.size() == expected.size(), "element count differs");
    auto value = expected.begin();
    for (double item : items) {
        close(item, *value++, absolute, relative);
    }
}

inline void gradient(const dl::Tensor& input, std::initializer_list<double> expected) {
    expect(input.grad().has_value(), "leaf gradient is absent");
    values(*input.grad(), expected);
}

template <class Function> int run(Function function) {
    try {
        function();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace test
