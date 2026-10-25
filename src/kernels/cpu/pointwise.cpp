#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "kernels/cpu/elementwise.h"
#include "kernels/cpu/vector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>

namespace dl::detail::cpu {
namespace {

// 高斯函数的固定系数在编译期求值，逐元素循环使用乘法。
template <class T> constexpr T inverse_sqrt2 = std::numbers::sqrt2_v<T> / T{2};
template <class T> constexpr T inverse_sqrt2pi = std::numbers::inv_sqrtpi_v<T> * inverse_sqrt2<T>;

template <class Function>
TensorData binary(const TensorData& a, const TensorData& b, Function function) {
    auto result = empty_data(broadcast_shape(a.layout.sizes, b.layout.sizes), accumulation_dtype(a.dtype));
    visit_storage(a.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        cpu::binary<T, T, A>(a, b, result, [&](T x, T y) {
            return function(static_cast<A>(x), static_cast<A>(y));
        });
    });
    return result;
}

template <class T> T sigmoid_value(T value) {
    const T exponential = std::exp(-std::abs(value));
    return value >= T{0} ? T{1} / (T{1} + exponential) : exponential / (T{1} + exponential);
}

template <class T> T relu_value(T value) { return value < T{0} ? T{0} : value; }

template <class T, class Apply> void dispatch_unary(Unary operation, Apply apply) {
    switch (operation) {
    case Unary::exp: return apply([](T value) { return std::exp(value); });
    case Unary::log: return apply([](T value) { return std::log(value); });
    case Unary::log1p: return apply([](T value) { return std::log1p(value); });
    case Unary::expm1: return apply([](T value) { return std::expm1(value); });
    case Unary::abs: return apply([](T value) { return std::abs(value); });
    case Unary::sin: return apply([](T value) { return std::sin(value); });
    case Unary::cos: return apply([](T value) { return std::cos(value); });
    case Unary::tan: return apply([](T value) { return std::tan(value); });
    case Unary::asin: return apply([](T value) { return std::asin(value); });
    case Unary::acos: return apply([](T value) { return std::acos(value); });
    case Unary::atan: return apply([](T value) { return std::atan(value); });
    case Unary::softplus: return apply([](T value) {
        return std::max(value, T{0}) + std::log1p(std::exp(-std::abs(value)));
    });
    case Unary::sqrt: return apply([](T value) { return std::sqrt(value); });
    case Unary::tanh: return apply([](T value) { return std::tanh(value); });
    case Unary::sigmoid: return apply([](T value) { return sigmoid_value(value); });
    case Unary::silu: return apply([](T value) { return value * sigmoid_value(value); });
    case Unary::relu: return apply([](T value) { return relu_value(value); });
    case Unary::gelu: break;
    }
    apply([](T value) {
        return T{0.5} * value * (T{1} + std::erf(value * inverse_sqrt2<T>));
    });
}

template <class T, class Apply> void dispatch_backward(Unary operation, Apply apply) {
    switch (operation) {
    case Unary::exp: return apply([](T grad, T value) { return grad * value; });
    case Unary::log: return apply([](T grad, T value) { return grad / value; });
    case Unary::log1p: return apply([](T grad, T value) { return grad / (T{1} + value); });
    case Unary::expm1: return apply([](T grad, T value) { return grad * std::exp(value); });
    case Unary::abs: return apply([](T grad, T value) {
        return grad * static_cast<T>((value > T{0}) - (value < T{0}));
    });
    case Unary::sin: return apply([](T grad, T value) { return grad * std::cos(value); });
    case Unary::cos: return apply([](T grad, T value) { return -grad * std::sin(value); });
    case Unary::tan: return apply([](T grad, T value) {
        const T cosine = std::cos(value);
        return grad / (cosine * cosine);
    });
    case Unary::asin: return apply([](T grad, T value) { return grad / std::sqrt(T{1} - value * value); });
    case Unary::acos: return apply([](T grad, T value) { return -grad / std::sqrt(T{1} - value * value); });
    case Unary::atan: return apply([](T grad, T value) { return grad / (T{1} + value * value); });
    case Unary::softplus: return apply([](T grad, T value) { return grad * sigmoid_value(value); });
    case Unary::sqrt: return apply([](T grad, T value) { return grad / (T{2} * value); });
    case Unary::tanh: return apply([](T grad, T value) { return grad * (T{1} - value * value); });
    case Unary::sigmoid: return apply([](T grad, T value) { return grad * (value * (T{1} - value)); });
    case Unary::silu: return apply([](T grad, T value) {
        const T gate = sigmoid_value(value);
        return grad * (gate * (T{1} + value * (T{1} - gate)));
    });
    // ReLU 的关闭分支直接给出零，开启分支保留上游梯度的数值。
    case Unary::relu: return apply([](T grad, T value) { return value <= T{0} ? T{0} : grad; });
    case Unary::gelu: break;
    }
    apply([](T grad, T value) {
        const T cdf = T{0.5} * (T{1} + std::erf(value * inverse_sqrt2<T>));
        const T density = std::exp(-T{0.5} * value * value) * inverse_sqrt2pi<T>;
        return grad * (cdf + value * density);
    });
}

} // namespace

TensorData full(Shape shape, double value, DType dtype, Device device) {
    auto result = empty_data(std::move(shape), dtype, device);
    visit_storage(dtype, [&]<class T> {
        std::fill_n(result.mutable_data<T>(), result.layout.numel(), static_cast<T>(value));
    });
    return result;
}

void copy_into(TensorData& target, const TensorData& source) {
    if (source.layout.is_contiguous() && target.layout.is_contiguous()) {
        std::memmove(target.mutable_address(), source.address(),
                     static_cast<std::size_t>(source.layout.numel()) * element_size(source.dtype));
        return;
    }
    IterationPlan<2> plan(source.layout.sizes, {source.layout.strides, target.layout.strides});
    visit_storage(source.dtype, [&]<class T> {
        auto* output = target.mutable_data<T>();
        const auto* input = source.data<T>();
        plan.for_each([&](Index, const auto& offsets) { output[offsets[1]] = input[offsets[0]]; });
    });
}

TensorData clone(const TensorData& input) {
    auto result = empty_data(input.layout.sizes, input.dtype);
    copy_into(result, input);
    return result;
}

TensorData cast(const TensorData& input, DType dtype) {
    auto result = empty_data(input.layout.sizes, dtype);
    IterationPlan<1> plan(input.layout.sizes, {input.layout.strides});
    visit_storage(input.dtype, [&]<class From> {
        visit_storage(dtype, [&]<class To> {
            const auto* source = input.data<From>();
            auto* target = result.mutable_data<To>();
            plan.for_each([&](Index index, const auto& offsets) {
                target[index] = static_cast<To>(source[offsets[0]]);
            });
        });
    });
    return result;
}

void axpy(TensorData& target, const TensorData& input, double alpha) {
    if (input.layout.is_contiguous() && target.layout.is_contiguous()) {
        const auto count = static_cast<std::size_t>(input.layout.numel());
        if (input.dtype == DType::f32) {
            dl_axpy_f32(count, static_cast<float>(alpha), input.data<float>(), target.mutable_data<float>());
        } else {
            dl_axpy_f64(count, alpha, input.data<double>(), target.mutable_data<double>());
        }
        return;
    }
    if (target.layout.is_contiguous()) {
        visit_floating(input.dtype, [&]<class T> {
            const auto factor = static_cast<T>(alpha);
            cpu::binary<T>(target, input, target, [factor](T value, T delta) { return value + factor * delta; });
        });
        return;
    }
    IterationPlan<2> plan(input.layout.sizes, {input.layout.strides, target.layout.strides});
    visit_floating(input.dtype, [&]<class T> {
        const auto* source = input.data<T>();
        auto* output = target.mutable_data<T>();
        const T factor = static_cast<T>(alpha);
        plan.for_each([&](Index, const auto& offsets) {
            output[offsets[1]] += factor * source[offsets[0]];
        });
    });
}

TensorData unary(const TensorData& input, Unary operation) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype));
    visit_storage(input.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        dispatch_unary<A>(operation, [&](auto kernel) { cpu::unary<T, A>(input, result, kernel); });
    });
    return result;
}

void unary_backward_into(TensorData& result, const TensorData& grad, const TensorData& saved,
                         Unary operation, bool accumulate) {
    visit_storage(saved.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        dispatch_backward<A>(operation, [&](auto kernel) {
            cpu::binary<A, T>(grad, saved, result, kernel, accumulate);
        });
    });
}

TensorData add(const TensorData& a, const TensorData& b) {
    return binary(a, b, [](auto x, auto y) { return x + y; });
}
TensorData cast_add(const TensorData& input, const TensorData& offset) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype));
    visit_storage(input.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        cpu::binary<T, A, A>(input, offset, result, [](T value, A bias) {
            return static_cast<A>(value) + bias;
        });
    });
    return result;
}
TensorData add_relu(const TensorData& a, const TensorData& b) {
    auto result = empty_data(broadcast_shape(a.layout.sizes, b.layout.sizes), accumulation_dtype(a.dtype));
    visit_storage(a.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        cpu::binary<T, T, A>(a, b, result, [](T x, T y) {
            return relu_value(static_cast<A>(x) + static_cast<A>(y));
        });
    });
    return result;
}
TensorData sub(const TensorData& a, const TensorData& b) {
    return binary(a, b, [](auto x, auto y) { return x - y; });
}
TensorData mul(const TensorData& a, const TensorData& b) {
    return binary(a, b, [](auto x, auto y) { return x * y; });
}
TensorData div(const TensorData& a, const TensorData& b) {
    return binary(a, b, [](auto x, auto y) { return x / y; });
}
TensorData pow(const TensorData& a, const TensorData& b) {
    return binary(a, b, [](auto x, auto y) { return std::pow(x, y); });
}

void scale_into(TensorData& result, const TensorData& input, double factor, bool accumulate) {
    visit_storage(input.dtype, [&]<class T> {
        using A = FloatAccumulator<T>;
        cpu::unary<T, A>(input, result, [factor](A value) { return value * static_cast<A>(factor); }, accumulate);
    });
}

TensorData where(const TensorData& mask, const TensorData& a, const TensorData& b) {
    auto shape = broadcast_shape(mask.layout.sizes, broadcast_shape(a.layout.sizes, b.layout.sizes));
    auto result = empty_data(shape, a.dtype);
    IterationPlan<3> plan(shape, {broadcast_strides(mask.layout, shape),
                                 broadcast_strides(a.layout, shape), broadcast_strides(b.layout, shape)});
    visit_storage(a.dtype, [&]<class T> {
        const auto* condition = mask.data<bool>();
        const auto* left = a.data<T>();
        const auto* right = b.data<T>();
        auto* output = result.mutable_data<T>();
        plan.for_each([&](Index index, const auto& offsets) {
            output[index] = condition[offsets[0]] ? left[offsets[1]] : right[offsets[2]];
        });
    });
    return result;
}

} // namespace dl::detail::cpu
