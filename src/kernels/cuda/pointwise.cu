#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "kernels/cuda/common.cuh"
#include "kernels/cuda/storage.cuh"

#include <cstdint>

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

struct CopyValue {
    template <class T> __device__ T operator()(T value, Index) const { return value; }
};

struct AddOffset {
    const float* values;
    IndexMap layout;
    __device__ float operator()(float value, Index index) const {
        return __fadd_rn(value, values[layout.offset(index)]);
    }
};

template <class From, class To, class Transform>
__global__ void copy_kernel(const From* input, To* output, Index count, IndexMap source, IndexMap target,
                            Transform transform) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[target.offset(index)] = transform(storage_cast<To>(input[source.offset(index)]), index);
    }
}

template <class T> struct alignas(4 * sizeof(T)) CopyPack { T values[4]; };

/// 连续数据按四个元素读写，类型转换仍逐项采用相同的舍入规则。
template <class From, class To, class Transform>
__device__ void packed_copy(const From* input, To* output, Index count, int block, int blocks,
                            Transform transform) {
    const Index thread = static_cast<Index>(block) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * blocks;
    for (Index group = thread; group < count / 4; group += stride) {
        const auto source = reinterpret_cast<const CopyPack<From>*>(input)[group];
        CopyPack<To> target;
#pragma unroll
        for (int item = 0; item < 4; ++item)
            target.values[item] = transform(storage_cast<To>(source.values[item]), group * 4 + item);
        reinterpret_cast<CopyPack<To>*>(output)[group] = target;
    }
    const Index tail = count / 4 * 4 + thread;
    if (tail < count) output[tail] = transform(storage_cast<To>(input[tail]), tail);
}

template <class From, class To, class Transform>
__global__ void packed_copy_kernel(const From* input, To* output, Index count, Transform transform) {
    packed_copy(input, output, count, blockIdx.x, gridDim.x, transform);
}

struct CastOperand {
    const float* input;
    void* output;
    Index count;
    int blocks;
};

/// 两个操作数按各自大小划分线程块，大小不同的矩阵也只提交一次转换。
template <class To> __global__ void cast_pair_kernel(CastOperand first, CastOperand second) {
    const bool right = blockIdx.x >= first.blocks;
    const auto operand = right ? second : first;
    const int block = static_cast<int>(blockIdx.x) - (right ? first.blocks : 0);
    packed_copy(operand.input, static_cast<To*>(operand.output), operand.count, block, operand.blocks, CopyValue{});
}

template <class T> bool packed(const TensorData& value) {
    return value.layout.is_contiguous()
        && reinterpret_cast<std::uintptr_t>(value.address()) % alignof(CopyPack<T>) == 0;
}

/// 分块转置使相邻线程连续读写，类型转换在写入时完成。
template <class From, class To, class Transform>
__global__ void transpose_copy_kernel(const From* input, To* output, Index rows, Index columns,
                                      Index row_tiles, Index column_tiles, Index tiles, Transform transform) {
    __shared__ From tile[32][33];
    const auto tiles_per_matrix = row_tiles * column_tiles;
    for (Index index = blockIdx.x; index < tiles; index += gridDim.x) {
        const auto batch = index / tiles_per_matrix;
        const auto row = (index % tiles_per_matrix / column_tiles) * 32;
        const auto column = (index % column_tiles) * 32;
        const auto base = batch * rows * columns;
        for (int offset = 0; offset < 32; offset += 8) {
            if (row + threadIdx.x < rows && column + threadIdx.y + offset < columns)
                tile[threadIdx.y + offset][threadIdx.x] =
                    input[base + (column + threadIdx.y + offset) * rows + row + threadIdx.x];
        }
        __syncthreads();
        for (int offset = 0; offset < 32; offset += 8) {
            if (row + threadIdx.y + offset < rows && column + threadIdx.x < columns) {
                const auto index = base + (row + threadIdx.y + offset) * columns + column + threadIdx.x;
                output[index] = transform(storage_cast<To>(tile[threadIdx.x][threadIdx.y + offset]), index);
            }
        }
        __syncthreads();
    }
}

template <class From, class To, class Transform = CopyValue>
void copy_elements(const TensorData& input, TensorData& output, cudaStream_t stream, Transform transform = {}) {
    const IndexMap source(input.layout);
    const auto count = input.layout.numel();
    if (packed<From>(input) && packed<To>(output)) {
        packed_copy_kernel<<<grid_size((count + 3) / 4), block_size, 0, stream>>>(
            input.data<From>(), output.mutable_data<To>(), count, transform);
        return;
    }
    // 合并连续轴后，矩阵转置和 NCHW/NHWC 转换都可表示为一批二维转置。
    if ((source.rank == 2 || source.rank == 3) && output.layout.is_contiguous()) {
        const auto rows = source.sizes[source.rank - 2];
        const auto columns = source.sizes[source.rank - 1];
        if (rows >= 32 && columns >= 32 && source.strides[source.rank - 2] == 1
            && source.strides[source.rank - 1] == rows
            && (source.rank == 2 || source.strides[0] == rows * columns)) {
            const auto row_tiles = (rows + 31) / 32;
            const auto column_tiles = (columns + 31) / 32;
            const auto tiles = (count / (rows * columns)) * row_tiles * column_tiles;
            transpose_copy_kernel<<<static_cast<int>(std::min(tiles, Index{4096})), dim3(32, 8), 0, stream>>>(
                input.data<From>(), output.mutable_data<To>(), rows, columns, row_tiles, column_tiles, tiles, transform);
            return;
        }
    }
    copy_kernel<<<grid_size(count), block_size, 0, stream>>>(input.data<From>(), output.mutable_data<To>(),
        count, source, IndexMap(output.layout), transform);
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

template <Binary Operation, class T>
__global__ void binary_kernel(const T* a, const T* b, float* output, Index count,
                               IndexMap left, IndexMap right) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const float x = storage_cast<float>(a[left.offset(index)]);
        const float y = storage_cast<float>(b[right.offset(index)]);
        if constexpr (Operation == Binary::add) output[index] = x + y;
        if constexpr (Operation == Binary::sub) output[index] = x - y;
        if constexpr (Operation == Binary::mul) output[index] = x * y;
        if constexpr (Operation == Binary::div) output[index] = x / y;
        if constexpr (Operation == Binary::pow) output[index] = powf(x, y);
    }
}

template <Binary Operation> TensorData binary(const TensorData& a, const TensorData& b) {
    auto shape = broadcast_shape(a.layout.sizes, b.layout.sizes);
    auto result = empty_data(shape, accumulation_dtype(a.dtype), a.device());
    const auto count = result.layout.numel();
    if (count != 0) {
        Execution execution(a);
        visit_storage(a.dtype, [&]<class T> {
            binary_kernel<Operation><<<grid_size(count), block_size, 0, execution.stream()>>>(
                a.data<T>(), b.data<T>(), result.mutable_data<float>(), count,
                IndexMap(shape, broadcast_strides(a.layout, shape)),
                IndexMap(shape, broadcast_strides(b.layout, shape)));
        });
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
            copy_elements<T, T>(source, target, execution.stream());
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
                copy_elements<From, To>(input, result, execution.stream());
            });
        });
        check(cudaGetLastError());
    }
    return result;
}
TensorData cast_add(const TensorData& input, const TensorData& offset) {
    auto result = empty_data(input.layout.sizes, accumulation_dtype(input.dtype), input.device());
    if (result.layout.numel() == 0) return result;
    Execution execution(input);
    const AddOffset transform{offset.data<float>(),
        IndexMap(result.layout.sizes, broadcast_strides(offset.layout, result.layout.sizes))};
    visit_storage(input.dtype, [&]<class T> {
        copy_elements<T, float>(input, result, execution.stream(), transform);
    });
    check(cudaGetLastError());
    return result;
}
std::array<TensorData, 2> cast_pair(const TensorData& first, const TensorData& second, DType dtype) {
    if (first.dtype != DType::f32 || second.dtype != DType::f32 || !is_low_precision(dtype)
        || !packed<float>(first) || !packed<float>(second))
        return {numeric::cast(first, dtype), numeric::cast(second, dtype)};
    std::array result{empty_data(first.layout.sizes, dtype, first.device()),
                      empty_data(second.layout.sizes, dtype, second.device())};
    const auto operand = [](const TensorData& input, TensorData& output) {
        const auto count = input.layout.numel();
        return CastOperand{input.data<float>(), output.mutable_address(), count,
                           count == 0 ? 0 : grid_size((count + 3) / 4)};
    };
    const auto left = operand(first, result[0]), right = operand(second, result[1]);
    if (left.blocks + right.blocks != 0) {
        Execution execution(first);
        visit_storage(dtype, [&]<class T> {
            cast_pair_kernel<T><<<left.blocks + right.blocks, block_size, 0, execution.stream()>>>(left, right);
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
