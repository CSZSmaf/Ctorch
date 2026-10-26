#include "kernels/cpu/ops.h"

#include "core/dispatch.h"
#include "kernels/cpu/moments.h"

#include <algorithm>
#include <cmath>

namespace dl::detail::cpu {
namespace {

// 连续存储直接使用线性下标，带步长的视图沿原布局寻址。
template <class Function>
void with_offsets(bool contiguous, Function function) {
    if (contiguous) function([](const Layout&, Index index) { return index; });
    else function([](const Layout& layout, Index index) { return layout.index_offset(index); });
}

} // namespace

void normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon,
    numeric::NormalizationKind kind, numeric::NormalizationResult& result) {
    const auto rows = input.layout.numel() / width;
    const bool centered = kind == numeric::NormalizationKind::layer;
    with_offsets(input.layout.is_contiguous() && (!weight || weight->layout.is_contiguous())
        && (!bias || bias->layout.is_contiguous()), [&](auto offset) {
        visit_storage(input.dtype, [&]<class T>() {
            using A = FloatAccumulator<T>;
            const auto* source = input.data<T>();
            const auto* scale = weight ? weight->data<A>() : nullptr;
            const auto* shift = bias ? bias->data<A>() : nullptr;
            auto* output = result.output.mutable_data<A>();
            auto* statistics = result.statistics ? result.statistics->mutable_data<double>() : nullptr;

            // 方差围绕均值计算，整行中间值只存在于标量中；原输入布局直接参与寻址。
            for (Index row = 0; row < rows; ++row) {
                const auto base = row * width;
                const auto [mean, variance] = moments(width, [&](Index col) {
                    return static_cast<A>(source[offset(input.layout, base + col)]);
                }, centered);
                const double inverse_std = 1.0 / std::sqrt(variance + epsilon);
                if (statistics) {
                    statistics[row] = mean;
                    statistics[rows + row] = inverse_std;
                }
                for (Index col = 0; col < width; ++col) {
                    double value = (static_cast<double>(source[offset(input.layout, base + col)]) - mean) * inverse_std;
                    if (scale) value *= scale[offset(weight->layout, col)];
                    if (shift) value += shift[offset(bias->layout, col)];
                    output[base + col] = static_cast<A>(value);
                }
            }
        });
    });
}

void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight, Index width,
    numeric::NormalizationKind kind, numeric::NormalizationGradients& result) {
    const auto rows = input.layout.numel() / width;
    const bool centered = kind == numeric::NormalizationKind::layer;
    with_offsets(input.layout.is_contiguous() && grad.layout.is_contiguous()
        && (!weight || weight->layout.is_contiguous()), [&](auto offset) {
        visit_storage(input.dtype, [&]<class T>() {
            using A = FloatAccumulator<T>;
            const auto* source = input.data<T>();
            const auto* upstream = grad.data<A>();
            const auto* moments = statistics.data<double>();
            const auto* scale = weight ? weight->data<A>() : nullptr;
            auto* dx = result.input ? result.input->mutable_data<A>() : nullptr;
            auto* dw = result.weight ? result.weight->mutable_data<A>() : nullptr;
            auto* db = result.bias ? result.bias->mutable_data<A>() : nullptr;
            if (dw && !result.accumulate[1]) std::fill_n(dw, width, A{0});
            if (db && !result.accumulate[2]) std::fill_n(db, width, A{0});
            const double inverse_width = 1.0 / static_cast<double>(width);

            for (Index row = 0; row < rows; ++row) {
                const auto base = row * width;
                const double mean = moments[row], inverse_std = moments[rows + row];
                double average = 0, projection = 0;
                if (dx) {
                    for (Index col = 0; col < width; ++col) {
                        const double z = (static_cast<double>(source[offset(input.layout, base + col)]) - mean) * inverse_std;
                        double value = upstream[offset(grad.layout, base + col)];
                        if (scale) value *= scale[offset(weight->layout, col)];
                        if (centered) average += value;
                        projection += value * z;
                    }
                    average *= inverse_width;
                    projection *= inverse_width;
                }

                // u = dy * weight；LayerNorm 去除均值方向和归一化方向，RMSNorm 只去除后者。
                // 同一次遍历完成逐行 dx 与跨行的仿射梯度累积，空 batch 自然留下零参数梯度。
                for (Index col = 0; col < width; ++col) {
                    const double z = (static_cast<double>(source[offset(input.layout, base + col)]) - mean) * inverse_std;
                    const A dy = upstream[offset(grad.layout, base + col)];
                    if (dx) {
                        const double value = scale ? static_cast<double>(dy) * scale[offset(weight->layout, col)] : dy;
                        const double contribution = (value - average - z * projection) * inverse_std;
                        dx[base + col] = static_cast<A>((result.accumulate[0] ? dx[base + col] : A{0}) + contribution);
                    }
                    if (dw) dw[col] = static_cast<A>(dw[col] + dy * z);
                    if (db) db[col] += dy;
                }
            }
        });
    });
}

} // namespace dl::detail::cpu
