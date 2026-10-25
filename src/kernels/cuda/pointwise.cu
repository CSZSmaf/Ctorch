#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

namespace dl::detail::cuda {
namespace {

/// 连续目标统一承接首次写入与分支累加，贡献在相加前完成 FP32 舍入。
__device__ void write_value(float& target, float value, bool accumulate) {
    target = accumulate ? __fadd_rn(target, value) : value;
}

template <class T> __global__ void fill_kernel(T* output, Index count, T value) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[index] = value;
    }
}

template <class From, class To>
__global__ void copy_kernel(const From* input, To* output, Index count, IndexMap source, IndexMap target) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[target.offset(index)] = storage_cast<To>(input[source.offset(index)]);
    }
}

template <bool Accumulate = false, class T, class Function>
__global__ void unary_kernel(const T* input, float* output, Index count, IndexMap layout,
                              Function function) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        write_value(output[index], function(storage_cast<float>(input[layout.offset(index)])), Accumulate);
    }
}

__device__ float sigmoid_value(float value) {
    const float exponential = expf(-fabsf(value));
    return value >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
}

template <Unary Operation> struct UnaryFunction {
    __device__ float operator()(float value) const {
        if constexpr (Operation == Unary::exp) return expf(value);
        if constexpr (Operation == Unary::log) return logf(value);
        if constexpr (Operation == Unary::log1p) return log1pf(value);
        if constexpr (Operation == Unary::expm1) return expm1f(value);
        if constexpr (Operation == Unary::abs) return fabsf(value);
        if constexpr (Operation == Unary::sin) return sinf(value);
        if constexpr (Operation == Unary::cos) return cosf(value);
        if constexpr (Operation == Unary::tan) return tanf(value);
        if constexpr (Operation == Unary::asin) return asinf(value);
        if constexpr (Operation == Unary::acos) return acosf(value);
        if constexpr (Operation == Unary::atan) return atanf(value);
        if constexpr (Operation == Unary::softplus) {
            return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
        }
        if constexpr (Operation == Unary::sqrt) return sqrtf(value);
        if constexpr (Operation == Unary::tanh) return tanhf(value);
        if constexpr (Operation == Unary::relu) return value < 0.0f ? 0.0f : value;
        if constexpr (Operation == Unary::gelu) {
            return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
        }
        if constexpr (Operation == Unary::sigmoid) return sigmoid_value(value);
        if constexpr (Operation == Unary::silu) return value * sigmoid_value(value);
    }

    // 一元 VJP 直接组合上游梯度与保存值，ReLU 沿分段选择对应贡献。
    __device__ float operator()(float grad, float value) const {
        if constexpr (Operation == Unary::exp) return grad * value;
        if constexpr (Operation == Unary::log) return grad / value;
        if constexpr (Operation == Unary::log1p) return grad / (1.0f + value);
        if constexpr (Operation == Unary::expm1) return grad * expf(value);
        if constexpr (Operation == Unary::abs) return grad * float((value > 0.0f) - (value < 0.0f));
        if constexpr (Operation == Unary::sin) return grad * cosf(value);
        if constexpr (Operation == Unary::cos) return -grad * sinf(value);
        if constexpr (Operation == Unary::tan) {
            const float cosine = cosf(value);
            return grad / (cosine * cosine);
        }
        if constexpr (Operation == Unary::asin) return grad / sqrtf(1.0f - value * value);
        if constexpr (Operation == Unary::acos) return -grad / sqrtf(1.0f - value * value);
        if constexpr (Operation == Unary::atan) return grad / (1.0f + value * value);
        if constexpr (Operation == Unary::softplus) return grad * sigmoid_value(value);
        if constexpr (Operation == Unary::sqrt) return grad / (2.0f * value);
        if constexpr (Operation == Unary::tanh) return grad * (1.0f - value * value);
        if constexpr (Operation == Unary::sigmoid) return grad * (value * (1.0f - value));
        if constexpr (Operation == Unary::silu) {
            const float gate = sigmoid_value(value);
            return grad * (gate * (1.0f + value * (1.0f - gate)));
        }
        if constexpr (Operation == Unary::relu) return value <= 0.0f ? 0.0f : grad;
        if constexpr (Operation == Unary::gelu) {
            const float cdf = 0.5f * (1.0f + erff(value * 0.7071067811865475f));
            const float density = expf(-0.5f * value * value) * 0.3989422804014327f;
            return grad * (cdf + value * density);
        }
    }
};

template <class T, class Function>
__global__ void unary_backward_kernel(const float* grad, const T* saved, float* output,
                                      Index count, IndexMap gradient, IndexMap value, Function function, bool accumulate) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        write_value(output[index],
            function(grad[gradient.offset(index)], storage_cast<float>(saved[value.offset(index)])), accumulate);
    }
}

template <class Function> void select_unary(Unary operation, Function function) {
    switch (operation) {
    case Unary::exp: return function.template operator()<Unary::exp>();
    case Unary::log: return function.template operator()<Unary::log>();
    case Unary::sqrt: return function.template operator()<Unary::sqrt>();
    case Unary::tanh: return function.template operator()<Unary::tanh>();
    case Unary::sigmoid: return function.template operator()<Unary::sigmoid>();
    case Unary::silu: return function.template operator()<Unary::silu>();
    case Unary::relu: return function.template operator()<Unary::relu>();
    case Unary::gelu: return function.template operator()<Unary::gelu>();
    case Unary::log1p: return function.template operator()<Unary::log1p>();
    case Unary::expm1: return function.template operator()<Unary::expm1>();
    case Unary::abs: return function.template operator()<Unary::abs>();
    case Unary::sin: return function.template operator()<Unary::sin>();
    case Unary::cos: return function.template operator()<Unary::cos>();
    case Unary::tan: return function.template operator()<Unary::tan>();
    case Unary::asin: return function.template operator()<Unary::asin>();
    case Unary::acos: return function.template operator()<Unary::acos>();
    case Unary::atan: return function.template operator()<Unary::atan>();
    case Unary::softplus: return function.template operator()<Unary::softplus>();
    }
}

enum class Binary { add, sub, mul, div, pow };

// 两个输入按各自布局读取，累积值直接进入 ReLU，输出同时作为反向保存值。
template <class T>
__global__ void add_relu_kernel(const T* a, const T* b, float* output, Index count,
                                IndexMap left, IndexMap right) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const float sum = storage_cast<float>(a[left.offset(index)])
            + storage_cast<float>(b[right.offset(index)]);
        output[index] = UnaryFunction<Unary::relu>{}(sum);
    }
}

template <Binary Operation>
__global__ void binary_kernel(const float* a, const float* b, float* output, Index count,
                               IndexMap left, IndexMap right) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const float x = a[left.offset(index)];
        const float y = b[right.offset(index)];
        if constexpr (Operation == Binary::add) output[index] = x + y;
        if constexpr (Operation == Binary::sub) output[index] = x - y;
        if constexpr (Operation == Binary::mul) output[index] = x * y;
        if constexpr (Operation == Binary::div) output[index] = x / y;
        if constexpr (Operation == Binary::pow) output[index] = powf(x, y);
    }
}

template <Binary Operation> TensorData binary(const TensorData& a, const TensorData& b) {
    auto shape = broadcast_shape(a.layout.sizes, b.layout.sizes);
    auto result = empty_data(shape, a.dtype, a.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(a);
        binary_kernel<Operation><<<grid_size(count), block_size, 0, execution.stream()>>>(
            a.data<float>(), b.data<float>(), result.mutable_data<float>(), count,
            IndexMap(shape, broadcast_strides(a.layout, shape)),
            IndexMap(shape, broadcast_strides(b.layout, shape)));
        check(cudaGetLastError());
    }
    return result;
}

struct Scale {
    float factor;
    __device__ float operator()(float value) const { return value * factor; }
};

template <class T>
__global__ void where_kernel(const bool* mask, const T* a, const T* b, T* output,
                              Index count, IndexMap condition, IndexMap left, IndexMap right) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[index] = mask[condition.offset(index)] ? a[left.offset(index)] : b[right.offset(index)];
    }
}

} // namespace

TensorData full(Shape shape, double value, DType dtype, Device device) {
    auto result = empty_data(std::move(shape), dtype, device);
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(result);
        cuda::visit_storage(dtype, [&]<class T> {
            fill_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                result.mutable_data<T>(), count, storage_cast<T>(value));
        });
        check(cudaGetLastError());
    }
    return result;
}

void copy_into(TensorData& target, const TensorData& source) {
    const auto count = source.layout.numel();
    if (count == 0) {
        return;
    }
    Execution execution(source);
    if (source.layout.is_contiguous() && target.layout.is_contiguous()) {
        check(cudaMemcpyAsync(target.mutable_address(), source.address(),
                               static_cast<std::size_t>(count) * element_size(source.dtype),
                               cudaMemcpyDeviceToDevice, execution.stream()));
    } else {
        cuda::visit_storage(source.dtype, [&]<class T> {
            copy_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                source.data<T>(), target.mutable_data<T>(), count,
                IndexMap(source.layout), IndexMap(target.layout));
        });
        check(cudaGetLastError());
    }
}
TensorData clone(const TensorData& input) {
    auto result = empty_data(input.layout.sizes, input.dtype, input.device());
    copy_into(result, input);
    return result;
}
TensorData cast(const TensorData& input, DType dtype) {
    auto result = empty_data(input.layout.sizes, dtype, input.device());
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        cuda::visit_storage(input.dtype, [&]<class From> {
            cuda::visit_storage(dtype, [&]<class To> {
                copy_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                    input.data<From>(), result.mutable_data<To>(), count,
                    IndexMap(input.layout), IndexMap(result.layout));
            });
        });
        check(cudaGetLastError());
    }
    return result;
}
TensorData unary(const TensorData& input, Unary operation) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype), input.device());
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        select_unary(operation, [&]<Unary Operation> {
            visit_storage(input.dtype, [&]<class T> {
                unary_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                    input.data<T>(), result.mutable_data<float>(), count, IndexMap(input.layout),
                    UnaryFunction<Operation>{});
            });
        });
        check(cudaGetLastError());
    }
    return result;
}
void unary_backward_into(TensorData& result, const TensorData& grad, const TensorData& saved,
                         Unary operation, bool accumulate) {
    const auto count = saved.layout.numel();
    if (count != 0) {
        Execution execution(saved);
        select_unary(operation, [&]<Unary Operation> {
            visit_storage(saved.dtype, [&]<class T> {
                unary_backward_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                    grad.data<float>(), saved.data<T>(), result.mutable_data<float>(), count,
                    IndexMap(grad.layout), IndexMap(saved.layout), UnaryFunction<Operation>{}, accumulate);
            });
        });
        check(cudaGetLastError());
    }
}
TensorData add(const TensorData& a, const TensorData& b) { return binary<Binary::add>(a, b); }
TensorData add_relu(const TensorData& a, const TensorData& b) {
    auto shape = broadcast_shape(a.layout.sizes, b.layout.sizes);
    auto result = empty_data(shape, accumulation_dtype(a.dtype), a.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(a);
        visit_storage(a.dtype, [&]<class T> {
            add_relu_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                a.data<T>(), b.data<T>(), result.mutable_data<float>(), count,
                IndexMap(shape, broadcast_strides(a.layout, shape)),
                IndexMap(shape, broadcast_strides(b.layout, shape)));
        });
        check(cudaGetLastError());
    }
    return result;
}
TensorData sub(const TensorData& a, const TensorData& b) { return binary<Binary::sub>(a, b); }
TensorData mul(const TensorData& a, const TensorData& b) { return binary<Binary::mul>(a, b); }
TensorData div(const TensorData& a, const TensorData& b) { return binary<Binary::div>(a, b); }
TensorData pow(const TensorData& a, const TensorData& b) { return binary<Binary::pow>(a, b); }
void scale_into(TensorData& result, const TensorData& input, double factor, bool accumulate) {
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        auto launch = [&]<bool Accumulate> {
            visit_storage(input.dtype, [&]<class T> {
                unary_kernel<Accumulate><<<grid_size(count), block_size, 0, execution.stream()>>>(
                    input.data<T>(), result.mutable_data<float>(), count, IndexMap(input.layout),
                    Scale{static_cast<float>(factor)});
            });
        };
        if (accumulate) launch.template operator()<true>();
        else launch.template operator()<false>();
        check(cudaGetLastError());
    }
}
TensorData where(const TensorData& mask, const TensorData& a, const TensorData& b) {
    auto shape = broadcast_shape(mask.layout.sizes, broadcast_shape(a.layout.sizes, b.layout.sizes));
    auto result = empty_data(shape, a.dtype, a.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(a);
        cuda::visit_storage(a.dtype, [&]<class T> {
            where_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(
                mask.data<bool>(), a.data<T>(), b.data<T>(), result.mutable_data<T>(), count,
                IndexMap(shape, broadcast_strides(mask.layout, shape)),
                IndexMap(shape, broadcast_strides(a.layout, shape)),
                IndexMap(shape, broadcast_strides(b.layout, shape)));
        });
        check(cudaGetLastError());
    }
    return result;
}

} // namespace dl::detail::cuda
