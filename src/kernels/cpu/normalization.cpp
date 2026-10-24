#include "kernels/cpu/ops.h"

#include "core/dispatch.h"

#include <algorithm>
#include <cmath>

namespace dl::detail::cpu {

void normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon,
    numeric::NormalizationKind kind, numeric::NormalizationResult& result) {
    const auto rows = input.layout.numel() / width;
    const bool centered = kind == numeric::NormalizationKind::layer;
    visit_storage(input.dtype, [&]<class T>() {
        using A = FloatAccumulator<T>;
        const auto* source = input.data<T>();
        const auto* scale = weight ? weight->data<A>() : nullptr;
        const auto* shift = bias ? bias->data<A>() : nullptr;
        auto* output = result.output.mutable_data<A>();
        auto* statistics = result.statistics ? result.statistics->mutable_data<A>() : nullptr;
        const A inverse_width = A{1} / static_cast<A>(width);

        // 方差围绕均值计算，整行中间值只存在于标量中；原输入布局直接参与寻址。
        for (Index row = 0; row < rows; ++row) {
            const auto base = row * width;
            A mean = 0;
            if (centered) {
                for (Index col = 0; col < width; ++col)
                    mean += static_cast<A>(source[input.layout.index_offset(base + col)]);
                mean *= inverse_width;
            }
            A squares = 0;
            for (Index col = 0; col < width; ++col) {
                const A value = static_cast<A>(source[input.layout.index_offset(base + col)]) - mean;
                squares += value * value;
            }
            const A inverse_std = A{1} / std::sqrt(squares * inverse_width + static_cast<A>(epsilon));
            if (statistics) {
                statistics[row] = mean;
                statistics[rows + row] = inverse_std;
            }
            for (Index col = 0; col < width; ++col) {
                A value = (static_cast<A>(source[input.layout.index_offset(base + col)]) - mean) * inverse_std;
                if (scale) value *= scale[weight->layout.index_offset(col)];
                if (shift) value += shift[bias->layout.index_offset(col)];
                output[base + col] = value;
            }
        }
    });
}

void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight, Index width,
    numeric::NormalizationKind kind, numeric::NormalizationGradients& result) {
    const auto rows = input.layout.numel() / width;
    const bool centered = kind == numeric::NormalizationKind::layer;
    visit_storage(input.dtype, [&]<class T>() {
        using A = FloatAccumulator<T>;
        const auto* source = input.data<T>();
        const auto* upstream = grad.data<A>();
        const auto* moments = statistics.data<A>();
        const auto* scale = weight ? weight->data<A>() : nullptr;
        auto* dx = result.input ? result.input->mutable_data<A>() : nullptr;
        auto* dw = result.weight ? result.weight->mutable_data<A>() : nullptr;
        auto* db = result.bias ? result.bias->mutable_data<A>() : nullptr;
        if (dw) std::fill_n(dw, width, A{0});
        if (db) std::fill_n(db, width, A{0});
        const A inverse_width = A{1} / static_cast<A>(width);

        for (Index row = 0; row < rows; ++row) {
            const auto base = row * width;
            const A mean = moments[row], inverse_std = moments[rows + row];
            A average = 0, projection = 0;
            if (dx) {
                for (Index col = 0; col < width; ++col) {
                    const A z = (static_cast<A>(source[input.layout.index_offset(base + col)]) - mean) * inverse_std;
                    A value = upstream[grad.layout.index_offset(base + col)];
                    if (scale) value *= scale[weight->layout.index_offset(col)];
                    if (centered) average += value;
                    projection += value * z;
                }
                average *= inverse_width;
                projection *= inverse_width;
            }

            // u = dy * weight；LayerNorm 去除均值方向和归一化方向，RMSNorm 只去除后者。
            // 同一次遍历完成逐行 dx 与跨行的仿射梯度累积，空 batch 自然留下零参数梯度。
            for (Index col = 0; col < width; ++col) {
                const A z = (static_cast<A>(source[input.layout.index_offset(base + col)]) - mean) * inverse_std;
                const A dy = upstream[grad.layout.index_offset(base + col)];
                if (dx) {
                    const A value = scale ? dy * scale[weight->layout.index_offset(col)] : dy;
                    dx[base + col] = (value - average - z * projection) * inverse_std;
                }
                if (dw) dw[col] += dy * z;
                if (db) db[col] += dy;
            }
        }
    });
}

} // namespace dl::detail::cpu
