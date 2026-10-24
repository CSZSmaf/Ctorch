#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/philox.h"
#include "dl/random.h"
#include "kernels/cpu/elementwise.h"

#include <algorithm>
#include <cmath>

namespace dl::detail::cpu {

TensorData random(Shape shape, numeric::Random operation, double first, double second,
                  TensorOptions options, Generator& generator) {
    auto result = empty_data(std::move(shape), options.dtype);
    const auto count = result.layout.numel();
    if (operation == numeric::Random::bernoulli) {
        std::bernoulli_distribution distribution(first);
        std::generate_n(result.mutable_data<bool>(), count, [&] { return distribution(generator.engine()); });
    } else {
        visit_floating(options.dtype, [&]<class T> {
            if (operation == numeric::Random::uniform) {
                std::uniform_real_distribution<T> distribution(static_cast<T>(first), static_cast<T>(second));
                std::generate_n(result.mutable_data<T>(), count, [&] { return distribution(generator.engine()); });
            } else {
                std::normal_distribution<T> distribution;
                std::generate_n(result.mutable_data<T>(), count, [&] { return distribution(generator.engine()); });
            }
        });
    }
    return result;
}

TensorData masked_scale(const TensorData& input, const TensorData& mask, double factor) {
    auto result = empty_data(input.layout.sizes, input.dtype);
    visit_floating(input.dtype, [&]<class T> {
        const auto scale = static_cast<T>(factor);
        cpu::binary<T, bool>(input, mask, result, [scale](T value, bool keep) {
            return keep ? value * scale : T{0};
        });
    });
    return result;
}

numeric::DropoutResult dropout(const TensorData& input, double probability, Generator& generator) {
    auto mask = cpu::random(input.layout.sizes, numeric::Random::bernoulli, 1.0 - probability, 0.0,
                       {.dtype = DType::boolean}, generator);
    auto output = masked_scale(input, mask, 1.0 / (1.0 - probability));
    return {std::move(output), std::move(mask)};
}

TensorData random_keyed(Shape shape, numeric::Random operation, double first, double second,
                        TensorOptions options, RandomKey key) {
    auto result = empty_data(std::move(shape), options.dtype);
    const auto count = result.layout.numel();
    using namespace random_bits;
    if (operation == numeric::Random::bernoulli) {
        const auto threshold = static_cast<std::uint64_t>(first * 0x1p53);
        auto* output = result.mutable_data<bool>();
        for (Index index = 0; index < count; ++index) {
            output[index] = selected(philox(key.seed, index, key.stream), threshold);
        }
    } else {
        visit_floating(options.dtype, [&]<class T> {
            auto generate = [&](auto sample) {
                auto* output = result.mutable_data<T>();
                for (Index index = 0; index < count; ++index) {
                    output[index] = sample(philox(key.seed, index, key.stream));
                }
            };
            if (operation == numeric::Random::uniform) {
                const auto low = static_cast<T>(first), high = static_cast<T>(second);
                generate([=](Block bits) { return low + (high - low) * unit<T>(bits.x, bits.y); });
            } else {
                generate([](Block bits) {
                    const auto radius = std::sqrt(T{-2} * std::log(unit<T>(bits.x, bits.y, true)));
                    return radius * std::cos(static_cast<T>(6.2831853071795864769) * unit<T>(bits.z, bits.w));
                });
            }
        });
    }
    return result;
}

numeric::DropoutResult dropout_keyed(const TensorData& input, double probability, RandomKey key) {
    auto output = empty_data(input.layout.sizes, input.dtype);
    auto mask = empty_data(input.layout.sizes, DType::boolean);
    const auto threshold = static_cast<std::uint64_t>((1.0 - probability) * 0x1p53);
    IterationPlan<1> plan(input.layout.sizes, {input.layout.strides});
    visit_floating(input.dtype, [&]<class T> {
        const auto scale = static_cast<T>(1.0 / (1.0 - probability));
        const auto* source = input.data<T>();
        auto* values = output.mutable_data<T>();
        auto* selected = mask.mutable_data<bool>();
        // key 直接按逻辑元素展开，掩码生成与前向缩放共用一次遍历。
        plan.for_each([&](Index index, const auto& offsets) {
            const bool keep = random_bits::selected(random_bits::philox(key.seed, index, key.stream), threshold);
            selected[index] = keep;
            values[index] = keep ? source[offsets[0]] * scale : T{0};
        });
    });
    return {std::move(output), std::move(mask)};
}

} // namespace dl::detail::cpu
