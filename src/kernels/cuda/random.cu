#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "core/dispatch.h"
#include "dl/random.h"
#include "kernels/cuda/common.cuh"
#include "core/philox.h"

namespace dl::detail::cuda {
namespace {

using random_bits::philox;
using random_bits::selected;
using random_bits::unit;

template <class T, numeric::Random Operation>
__global__ void random_kernel(T* output, Index count, RandomKey key, T first, T second) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto bits = philox(key.seed, index, key.stream);
        if constexpr (Operation == numeric::Random::uniform) {
            output[index] = first + (second - first) * unit<T>(bits.x, bits.y);
        } else {
            // Box-Muller 的半径使用 (0,1]，角度使用 [0,1)，每个元素独立确定。
            const auto radius = sqrt(T{-2} * log(unit<T>(bits.x, bits.y, true)));
            output[index] = radius * cos(static_cast<T>(6.2831853071795864769) * unit<T>(bits.z, bits.w));
        }
    }
}

__global__ void bernoulli_kernel(bool* output, Index count, RandomKey key, std::uint64_t threshold) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto bits = philox(key.seed, index, key.stream);
        output[index] = selected(bits, threshold);
    }
}

__global__ void dropout_kernel(const float* input, float* output, bool* mask, Index count,
                               IndexMap layout, RandomKey key, std::uint64_t threshold, float scale) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        const auto bits = philox(key.seed, index, key.stream);
        const bool keep = selected(bits, threshold);
        mask[index] = keep;
        output[index] = keep ? input[layout.offset(index)] * scale : 0.0f;
    }
}

__global__ void mask_kernel(const float* input, const bool* mask, float* output, Index count,
                            IndexMap input_layout, IndexMap mask_layout, float scale) {
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<Index>(blockDim.x) * gridDim.x) {
        output[index] = mask[mask_layout.offset(index)] ? input[input_layout.offset(index)] * scale : 0.0f;
    }
}

} // namespace

TensorData random(Shape shape, numeric::Random operation, double first, double second,
                  TensorOptions options, Generator& generator) {
    const RandomKey key{numel(shape) == 0 ? 0 : generator.engine()()};
    return random_keyed(std::move(shape), operation, first, second, options, key);
}

TensorData random_keyed(Shape shape, numeric::Random operation, double first, double second,
                        TensorOptions options, RandomKey key) {
    auto result = empty_data(std::move(shape), options.dtype, options.device);
    const auto count = result.layout.numel();
    if (count == 0) return result;
    Execution execution(result);
    if (operation == numeric::Random::bernoulli) {
        bernoulli_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(result.mutable_data<bool>(),
            count, key, static_cast<std::uint64_t>(first * 0x1p53));
    } else {
        visit_floating(options.dtype, [&]<class T> {
            if (operation == numeric::Random::uniform) {
                random_kernel<T, numeric::Random::uniform><<<grid_size(count), block_size, 0, execution.stream()>>>(
                    result.mutable_data<T>(), count, key, static_cast<T>(first), static_cast<T>(second));
            } else {
                random_kernel<T, numeric::Random::normal><<<grid_size(count), block_size, 0, execution.stream()>>>(
                    result.mutable_data<T>(), count, key, T{0}, T{1});
            }
        });
    }
    check(cudaGetLastError());
    return result;
}

numeric::DropoutResult dropout(const TensorData& input, double probability, Generator& generator) {
    const RandomKey key{input.layout.numel() == 0 ? 0 : generator.engine()()};
    return dropout_keyed(input, probability, key);
}

numeric::DropoutResult dropout_keyed(const TensorData& input, double probability, RandomKey key) {
    auto output = empty_data(input.layout.sizes, input.dtype, input.device());
    auto mask = empty_data(input.layout.sizes, DType::boolean, input.device());
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        dropout_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(input.data<float>(),
            output.mutable_data<float>(), mask.mutable_data<bool>(), count, IndexMap(input.layout),
            key, static_cast<std::uint64_t>((1.0 - probability) * 0x1p53),
            static_cast<float>(1.0 / (1.0 - probability)));
        check(cudaGetLastError());
    }
    return {std::move(output), std::move(mask)};
}

TensorData masked_scale(const TensorData& input, const TensorData& mask, double factor) {
    auto output = empty_data(input.layout.sizes, input.dtype, input.device());
    const auto count = input.layout.numel();
    if (count != 0) {
        Execution execution(input);
        mask_kernel<<<grid_size(count), block_size, 0, execution.stream()>>>(input.data<float>(), mask.data<bool>(),
            output.mutable_data<float>(), count, IndexMap(input.layout), IndexMap(mask.layout), static_cast<float>(factor));
        check(cudaGetLastError());
    }
    return output;
}

} // namespace dl::detail::cuda
