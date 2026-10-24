#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "core/resampling.h"

namespace dl::detail::cpu {
namespace {

template <bool Backward>
void adaptive_pool(const TensorData& source, TensorData& result) {
    const auto geometry = resampling_geometry(Backward ? result.layout.sizes : source.layout.sizes,
        Backward ? source.layout.sizes : result.layout.sizes);
    const auto count = Backward ? source.layout.numel() : result.layout.numel();
    visit_floating(source.dtype, [&]<class T> {
        const auto* input = source.data<T>();
        auto* output = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            const auto region = geometry.window(index);
            T value{};
            if constexpr (Backward) value = input[source.layout.index_offset(index)] / static_cast<T>(region.count());
            for (Index y = region.first_y; y < region.last_y; ++y) {
                for (Index x = region.first_x; x < region.last_x; ++x) {
                    const auto pixel = geometry.offset(region.plane, y, x);
                    if constexpr (Backward) output[pixel] += value;
                    else value += input[source.layout.index_offset(pixel)];
                }
            }
            if constexpr (!Backward) output[index] = value / static_cast<T>(region.count());
        }
    });
}

template <bool Backward>
void resize(const TensorData& source, TensorData& result, InterpolationMode mode, bool align_corners) {
    const auto geometry = resampling_geometry(Backward ? result.layout.sizes : source.layout.sizes,
        Backward ? source.layout.sizes : result.layout.sizes);
    const auto count = Backward ? source.layout.numel() : result.layout.numel();
    visit_floating(source.dtype, [&]<class T> {
        const auto* input = source.data<T>();
        auto* output = result.mutable_data<T>();
        for (Index index = 0; index < count; ++index) {
            const auto point = geometry.point(index, mode, align_corners);
            T value{};
            if constexpr (Backward) value = input[source.layout.index_offset(index)];
            for (int item = 0; item < point.count; ++item) {
                const T weight = static_cast<T>(point.weights[item]);
                if constexpr (Backward) output[point.indices[item]] += value * weight;
                else value += input[source.layout.index_offset(point.indices[item])] * weight;
            }
            if constexpr (!Backward) output[index] = value;
        }
    });
}

} // namespace

TensorData adaptive_avg_pool2d(const TensorData& input, Size2 output_size) {
    auto result = empty_data(resampling_shape(input.layout.sizes, output_size), input.dtype);
    adaptive_pool<false>(input, result);
    return result;
}
void adaptive_avg_pool2d_backward_into(TensorData& target, const TensorData& grad) {
    adaptive_pool<true>(grad, target);
}
TensorData interpolate(const TensorData& input, Size2 output_size, InterpolationMode mode, bool align_corners) {
    auto result = empty_data(resampling_shape(input.layout.sizes, output_size), input.dtype);
    resize<false>(input, result, mode, align_corners);
    return result;
}
void interpolate_backward_into(TensorData& target, const TensorData& grad, InterpolationMode mode, bool align_corners) {
    resize<true>(grad, target, mode, align_corners);
}

} // namespace dl::detail::cpu
