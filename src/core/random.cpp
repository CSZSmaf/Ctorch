#include "dl/random.h"

#include "core/tensor_impl.h"
#include "core/philox.h"
#include "ops/numeric.h"

#include <algorithm>
#include <bit>
#include <locale>
#include <sstream>

namespace dl {

RandomKey fold_in(RandomKey key, std::uint64_t label) noexcept {
    const auto bits = detail::random_bits::philox(key.seed, label, key.stream);
    return {static_cast<std::uint64_t>(bits.x) | (static_cast<std::uint64_t>(bits.y) << 32),
        static_cast<std::uint64_t>(bits.z) | (static_cast<std::uint64_t>(bits.w) << 32)};
}

std::array<RandomKey, 2> split(RandomKey key) noexcept { return {fold_in(key, 0), fold_in(key, 1)}; }

std::vector<RandomKey> split(RandomKey key, std::size_t count) {
    std::vector<RandomKey> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) result.push_back(fold_in(key, i));
    return result;
}

Tensor key_tensor(RandomKey key) {
    const std::array<Index, 2> values{std::bit_cast<Index>(key.seed), std::bit_cast<Index>(key.stream)};
    return from_span<Index>(values, {2});
}

RandomKey key_from_tensor(const Tensor& tensor) {
    const auto* values = tensor.data<Index>();
    return {std::bit_cast<std::uint64_t>(values[0]), std::bit_cast<std::uint64_t>(values[1])};
}

Generator& default_generator() {
    thread_local Generator generator;
    return generator;
}

void manual_seed(std::uint64_t seed) { default_generator().seed(seed); }
void Generator::shuffle(std::span<Index> values) { std::shuffle(values.begin(), values.end(), engine_); }

std::string Generator::state() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << engine_;
    return output.str();
}

void Generator::load_state(std::string_view state) {
    std::istringstream input{std::string(state)};
    input.imbue(std::locale::classic());
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input >> engine_;
}

Tensor uniform(Shape shape, double low, double high, TensorOptions options, Generator& generator) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::uniform,
        low, high, options, generator), options.requires_grad);
}

Tensor randn(Shape shape, TensorOptions options, Generator& generator) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::normal,
        0.0, 1.0, options, generator), options.requires_grad);
}

Tensor bernoulli(Shape shape, double probability, Generator& generator, Device device) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::bernoulli,
        probability, 0.0, {.dtype = DType::boolean, .device = device}, generator));
}

Tensor uniform(RandomKey key, Shape shape, double low, double high, TensorOptions options) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::uniform,
        low, high, options, key), options.requires_grad);
}

Tensor randn(RandomKey key, Shape shape, TensorOptions options) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::normal,
        0.0, 1.0, options, key), options.requires_grad);
}

Tensor bernoulli(RandomKey key, Shape shape, double probability, Device device) {
    return detail::make_tensor(detail::numeric::random(std::move(shape), detail::numeric::Random::bernoulli,
        probability, 0.0, {.dtype = DType::boolean, .device = device}, key));
}

} // namespace dl
