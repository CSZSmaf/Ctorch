#include "dl/ops.h"

#include "autograd/record.h"
#include "ops/numeric.h"

namespace dl {
namespace {

using detail::data_of;
using detail::gradient_edge;
using detail::record;
namespace numeric = detail::numeric;

Tensor scalar_like(const Tensor& input, double value) {
    return full({}, value, {.dtype = input.dtype(), .device = input.device()});
}

Tensor comparison(const Tensor& a, const Tensor& b, numeric::Compare operation) {
    return detail::make_tensor(numeric::compare(data_of(a), data_of(b), operation));
}

Tensor unary(const Tensor& input, numeric::Unary operation) {
    auto result = numeric::unary(data_of(input), operation);
    // 这些函数可由输出直接得到导数；其余函数保存输入，保留符号或局部数值精度。
    const bool saves_output = operation == numeric::Unary::exp || operation == numeric::Unary::sqrt
        || operation == numeric::Unary::tanh || operation == numeric::Unary::sigmoid
        || operation == numeric::Unary::relu;
    auto saved = saves_output ? result : data_of(input);
    return record(std::move(result), {gradient_edge(input)},
        [saved = std::move(saved), operation](const auto& grad, auto& sink, const auto& edges) {
            if (!sink.needs(edges[0])) return;
            auto target = sink.target(edges[0], saved.layout.sizes, saved.dtype, saved.device());
            numeric::unary_backward_into(target.value, grad, saved, operation, target.accumulate);
        });
}

Tensor add_or_subtract(const Tensor& a, const Tensor& b, bool subtract) {
    auto result = subtract ? numeric::sub(data_of(a), data_of(b)) : numeric::add(data_of(a), data_of(b));
    return record(std::move(result), {gradient_edge(a), gradient_edge(b)},
        [a_shape = a.shape(), b_shape = b.shape(), subtract]
        (const auto& grad, auto& sink, const auto& edges) {
            if (sink.needs(edges[0])) {
                sink.add_reduced(edges[0], grad, a_shape);
            }
            if (sink.needs(edges[1])) {
                if (subtract) {
                    auto contribution = numeric::sum_to_shape(grad, b_shape);
                    sink.add_scaled(edges[1], contribution, -1.0);
                } else {
                    sink.add_reduced(edges[1], grad, b_shape);
                }
            }
        });
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) { return add_or_subtract(a, b, false); }
Tensor add_relu(const Tensor& a, const Tensor& b) {
    auto result = numeric::add_relu(data_of(a), data_of(b));
    auto saved = result;
    return record(std::move(result), {gradient_edge(a), gradient_edge(b)},
        [saved = std::move(saved), a_shape = a.shape(), b_shape = b.shape()]
        (const auto& grad, auto& sink, const auto& edges) {
            // 同一份分段导数分别沿两条广播关系还原，重复输入仍累计两次贡献。
            const auto contribution = numeric::unary_backward(grad, saved, numeric::Unary::relu);
            if (sink.needs(edges[0])) sink.add_reduced(edges[0], contribution, a_shape);
            if (sink.needs(edges[1])) sink.add_reduced(edges[1], contribution, b_shape);
        });
}
Tensor sub(const Tensor& a, const Tensor& b) { return add_or_subtract(a, b, true); }

Tensor mul(const Tensor& a, const Tensor& b) {
    std::vector edges{gradient_edge(a), gradient_edge(b)};
    std::optional<detail::SavedTensor> saved_a = edges[1].node ? std::optional{data_of(a)} : std::nullopt;
    std::optional<detail::SavedTensor> saved_b = edges[0].node ? std::optional{data_of(b)} : std::nullopt;
    return record(numeric::mul(data_of(a), data_of(b)), std::move(edges),
        [saved_a = std::move(saved_a), saved_b = std::move(saved_b),
         a_shape = a.shape(), b_shape = b.shape()](const auto& grad, auto& sink, const auto& inputs) {
            if (sink.needs(inputs[0])) {
                sink.add_owned(inputs[0], numeric::sum_to_shape(numeric::mul(grad, *saved_b), a_shape));
            }
            if (sink.needs(inputs[1])) {
                sink.add_owned(inputs[1], numeric::sum_to_shape(numeric::mul(grad, *saved_a), b_shape));
            }
        });
}

Tensor div(const Tensor& a, const Tensor& b) {
    std::vector edges{gradient_edge(a), gradient_edge(b)};
    auto denominator = data_of(b);
    auto result = numeric::div(data_of(a), denominator);
    std::optional<detail::SavedTensor> quotient = edges[1].node ? std::optional{result} : std::nullopt;
    return record(std::move(result), std::move(edges),
        [quotient = std::move(quotient), denominator, a_shape = a.shape(), b_shape = b.shape()]
        (const auto& grad, auto& sink, const auto& inputs) {
            if (sink.needs(inputs[0])) {
                sink.add_owned(inputs[0], numeric::sum_to_shape(numeric::div(grad, denominator), a_shape));
            }
            if (sink.needs(inputs[1])) {
                // dB = -g * (Y / B)，保存前向商后通过相除表达分母的梯度系数。
                auto coefficient = numeric::div(*quotient, denominator);
                auto contribution = numeric::scale(numeric::mul(grad, coefficient), -1.0);
                sink.add_owned(inputs[1], numeric::sum_to_shape(contribution, b_shape));
            }
        });
}

Tensor neg(const Tensor& input) {
    return record(numeric::scale(data_of(input), -1.0), {gradient_edge(input)},
        [](const auto& grad, auto& sink, const auto& edges) {
            sink.add_scaled(edges[0], grad, -1.0);
        });
}

Tensor pow(const Tensor& base, const Tensor& exponent) {
    auto result = numeric::pow(data_of(base), data_of(exponent));
    auto saved = result;
    return record(std::move(result), {gradient_edge(base), gradient_edge(exponent)},
        [a = data_of(base), b = data_of(exponent), saved = std::move(saved)]
        (const auto& grad, auto& sink, const auto& edges) {
            auto zero = numeric::full({}, 0.0, a.dtype, a.device());
            if (sink.needs(edges[0])) {
                auto one = numeric::full({}, 1.0, a.dtype, a.device());
                auto slope = numeric::mul(b, numeric::pow(a, numeric::sub(b, one)));
                auto contribution = numeric::where(numeric::compare(b, zero, numeric::Compare::eq),
                    zero, numeric::mul(grad, slope));
                sink.add_owned(edges[0], numeric::sum_to_shape(contribution, a.layout.sizes));
            }
            if (sink.needs(edges[1])) {
                auto slope = numeric::mul(saved, numeric::unary(a, numeric::Unary::log));
                auto at_zero = numeric::where(numeric::compare(b, zero, numeric::Compare::gt), zero, slope);
                slope = numeric::where(numeric::compare(a, zero, numeric::Compare::eq), at_zero, slope);
                sink.add_owned(edges[1], numeric::sum_to_shape(numeric::mul(grad, slope), b.layout.sizes));
            }
        });
}

Tensor pow(const Tensor& base, double exponent) { return pow(base, scalar_like(base, exponent)); }
Tensor pow(double base, const Tensor& exponent) { return pow(scalar_like(exponent, base), exponent); }

Tensor exp(const Tensor& input) { return unary(input, numeric::Unary::exp); }
Tensor log(const Tensor& input) { return unary(input, numeric::Unary::log); }
Tensor log1p(const Tensor& input) { return unary(input, numeric::Unary::log1p); }
Tensor expm1(const Tensor& input) { return unary(input, numeric::Unary::expm1); }
Tensor abs(const Tensor& input) { return unary(input, numeric::Unary::abs); }
Tensor sin(const Tensor& input) { return unary(input, numeric::Unary::sin); }
Tensor cos(const Tensor& input) { return unary(input, numeric::Unary::cos); }
Tensor tan(const Tensor& input) { return unary(input, numeric::Unary::tan); }
Tensor asin(const Tensor& input) { return unary(input, numeric::Unary::asin); }
Tensor acos(const Tensor& input) { return unary(input, numeric::Unary::acos); }
Tensor atan(const Tensor& input) { return unary(input, numeric::Unary::atan); }
Tensor sqrt(const Tensor& input) { return unary(input, numeric::Unary::sqrt); }
Tensor tanh(const Tensor& input) { return unary(input, numeric::Unary::tanh); }
Tensor sigmoid(const Tensor& input) { return unary(input, numeric::Unary::sigmoid); }
Tensor relu(const Tensor& input) { return unary(input, numeric::Unary::relu); }
Tensor gelu(const Tensor& input) { return unary(input, numeric::Unary::gelu); }
Tensor softplus(const Tensor& input) { return unary(input, numeric::Unary::softplus); }
Tensor silu(const Tensor& input) { return unary(input, numeric::Unary::silu); }
Tensor leaky_relu(const Tensor& input, double negative_slope) {
    return where(gt(input, scalar_like(input, 0.0)), input, input * negative_slope);
}

Tensor clamp_min(const Tensor& input, double minimum) {
    auto bound = scalar_like(input, minimum);
    return where(lt(input, bound), bound, input);
}
Tensor clamp_max(const Tensor& input, double maximum) {
    auto bound = scalar_like(input, maximum);
    return where(gt(input, bound), bound, input);
}
Tensor clamp(const Tensor& input, double minimum, double maximum) {
    return clamp_max(clamp_min(input, minimum), maximum);
}

Tensor eq(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::eq); }
Tensor ne(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::ne); }
Tensor lt(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::lt); }
Tensor le(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::le); }
Tensor gt(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::gt); }
Tensor ge(const Tensor& a, const Tensor& b) { return comparison(a, b, numeric::Compare::ge); }

Tensor where(const Tensor& mask, const Tensor& a, const Tensor& b) {
    return record(numeric::where(data_of(mask), data_of(a), data_of(b)),
        {gradient_edge(a), gradient_edge(b)},
        [condition = data_of(mask), a_shape = a.shape(), b_shape = b.shape()]
        (const auto& grad, auto& sink, const auto& edges) {
            auto zero = numeric::full({}, 0.0, grad.dtype, grad.device());
            if (sink.needs(edges[0])) {
                sink.add_owned(edges[0], numeric::sum_to_shape(numeric::where(condition, grad, zero), a_shape));
            }
            if (sink.needs(edges[1])) {
                sink.add_owned(edges[1], numeric::sum_to_shape(numeric::where(condition, zero, grad), b_shape));
            }
        });
}

Tensor where(const Tensor& mask, const Tensor& input, double other) {
    return where(mask, input, scalar_like(input, other));
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator-(const Tensor& input) { return neg(input); }
Tensor operator+(const Tensor& input, double value) { return add(input, scalar_like(input, value)); }
Tensor operator-(const Tensor& input, double value) { return sub(input, scalar_like(input, value)); }
Tensor operator*(const Tensor& input, double value) {
    // 常量直接传入缩放内核，前向与反向都只生成结果张量。
    return record(numeric::scale(data_of(input), value), {gradient_edge(input)},
        [value](const auto& grad, auto& sink, const auto& edges) {
            sink.add_scaled(edges[0], grad, value);
        });
}
Tensor operator/(const Tensor& input, double value) { return div(input, scalar_like(input, value)); }
Tensor operator+(double value, const Tensor& input) { return input + value; }
Tensor operator-(double value, const Tensor& input) { return sub(scalar_like(input, value), input); }
Tensor operator*(double value, const Tensor& input) { return input * value; }
Tensor operator/(double value, const Tensor& input) { return div(scalar_like(input, value), input); }

} // namespace dl
