#include "kernels/cpu/ops.h"
#include "core/dispatch.h"

#include <cmath>

namespace dl::detail::cpu {

void batch_normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, std::optional<numeric::BatchNormalizationState>& state,
    bool training, double momentum, double epsilon, numeric::NormalizationResult& result) {
    const auto channels = input.layout.sizes[1];
    Index spatial = 1;
    for (std::size_t i = 2; i < input.layout.sizes.size(); ++i) spatial *= input.layout.sizes[i];
    const auto count = input.layout.sizes[0] * spatial;
    visit_storage(input.dtype, [&]<class T>() {
        using A = FloatAccumulator<T>;
        const auto* x = input.data<T>();
        auto* y = result.output.mutable_data<A>();
        auto* statistics = result.statistics->mutable_data<A>();
        auto* running_mean = state ? state->mean.mutable_data<A>() : nullptr;
        auto* running_variance = state ? state->variance.mutable_data<A>() : nullptr;
        for (Index c = 0; c < channels; ++c) {
            auto index = [&](Index i) { return (i / spatial * channels + c) * spatial + i % spatial; };
            A mean = 0, variance = 0;
            if (training || !state) {
                for (Index i = 0; i < count; ++i) mean += static_cast<A>(x[input.layout.index_offset(index(i))]);
                mean /= static_cast<A>(count);
                for (Index i = 0; i < count; ++i) {
                    const A centered = static_cast<A>(x[input.layout.index_offset(index(i))]) - mean;
                    variance += centered * centered;
                }
                variance /= static_cast<A>(count);
                if (state) {
                    auto& m = running_mean[state->mean.layout.index_offset(c)];
                    auto& v = running_variance[state->variance.layout.index_offset(c)];
                    m = static_cast<A>(1.0 - momentum) * m + static_cast<A>(momentum) * mean;
                    v = static_cast<A>(1.0 - momentum) * v
                        + static_cast<A>(momentum * count / (count - 1)) * variance;
                }
            } else {
                mean = running_mean[state->mean.layout.index_offset(c)];
                variance = running_variance[state->variance.layout.index_offset(c)];
            }
            const A inverse = A{1} / std::sqrt(variance + static_cast<A>(epsilon));
            statistics[c] = mean;
            statistics[channels + c] = inverse;
            const A scale = weight ? weight->data<A>()[weight->layout.index_offset(c)] : A{1};
            const A shift = bias ? bias->data<A>()[bias->layout.index_offset(c)] : A{0};
            for (Index i = 0; i < count; ++i) {
                const auto position = index(i);
                y[position] = ((static_cast<A>(x[input.layout.index_offset(position)]) - mean) * inverse) * scale + shift;
            }
        }
    });
}

void batch_normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    bool batch_statistics, numeric::NormalizationGradients& result) {
    const auto channels = input.layout.sizes[1];
    Index spatial = 1;
    for (std::size_t i = 2; i < input.layout.sizes.size(); ++i) spatial *= input.layout.sizes[i];
    const auto count = input.layout.sizes[0] * spatial;
    visit_storage(input.dtype, [&]<class T>() {
        using A = FloatAccumulator<T>;
        const auto* x = input.data<T>();
        const auto* dy = grad.data<A>();
        const auto* stats = statistics.data<A>();
        auto* dx = result.input ? result.input->mutable_data<A>() : nullptr;
        auto* dw = result.weight ? result.weight->mutable_data<A>() : nullptr;
        auto* db = result.bias ? result.bias->mutable_data<A>() : nullptr;
        for (Index c = 0; c < channels; ++c) {
            auto index = [&](Index i) { return (i / spatial * channels + c) * spatial + i % spatial; };
            const A mean = stats[c], inverse = stats[channels + c];
            A total = 0, projection = 0;
            if (dw || db || (dx && batch_statistics)) {
                for (Index i = 0; i < count; ++i) {
                    const auto position = index(i);
                    const A g = dy[grad.layout.index_offset(position)];
                    total += g;
                    if (dw || (dx && batch_statistics))
                        projection += g * ((static_cast<A>(x[input.layout.index_offset(position)]) - mean) * inverse);
                }
            }
            if (dw) dw[c] = (result.accumulate[1] ? dw[c] : A{0}) + projection;
            if (db) db[c] = (result.accumulate[2] ? db[c] : A{0}) + total;
            if (!dx) continue;
            const A factor = inverse * (weight ? weight->data<A>()[weight->layout.index_offset(c)] : A{1});
            for (Index i = 0; i < count; ++i) {
                const auto position = index(i);
                A value = dy[grad.layout.index_offset(position)];
                if (batch_statistics) {
                    const A normalized = (static_cast<A>(x[input.layout.index_offset(position)]) - mean) * inverse;
                    value = value - total / static_cast<A>(count) - normalized * (projection / static_cast<A>(count));
                }
                dx[position] = (result.accumulate[0] ? dx[position] : A{0}) + value * factor;
            }
        }
    });
}

} // namespace dl::detail::cpu
