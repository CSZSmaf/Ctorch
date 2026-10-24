#include "dl/ops.h"

#include "autograd/record.h"
#include "dl/amp.h"
#include "ops/numeric.h"

namespace dl {
namespace {

namespace numeric = detail::numeric;

DType matrix_compute_dtype(const Tensor& input) {
    return input.dtype() == DType::f32 && is_autocast_enabled(input.device().type)
        ? autocast_dtype(input.device().type) : input.dtype();
}

void matmul_gradient(const detail::Edge& edge, const detail::TensorData& left,
    const detail::TensorData& right, const Shape& shape, detail::GradSink& sink) {
    auto product_shape = numeric::matmul_shape(left, right);
    if (detail::numel(product_shape) != detail::numel(shape)) {
        // 广播产生多个矩阵贡献时保留批量 GEMM，再按原输入的批维求和。
        sink.add_reduced(edge, numeric::matmul(left, right), shape);
        return;
    }
    auto target = sink.target(edge, shape, left.dtype, left.device());
    // 目标临时采用乘积的形状，合并行轴或单例批维都继续写入同一连续存储。
    auto output = numeric::reshape(target.value, std::move(product_shape));
    numeric::matmul_into(output, left, right, target.accumulate);
}

} // namespace

Tensor linear(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias) {
    const auto compute_dtype = matrix_compute_dtype(input);
    const auto shape = std::span<const Index>(input.shape());
    const auto rows = detail::numel(shape.first(shape.size() - 1));
    const auto columns = weight.size(0);
    // 所有前导维组成同一个行轴，连续输入共享存储，一般布局按 reshape 规则准备。
    auto matrix = numeric::reshape(numeric::cast(detail::data_of(input), compute_dtype), {rows, shape.back()});
    auto weights = numeric::cast(detail::data_of(weight), compute_dtype);
    auto output_shape = input.shape();
    output_shape.back() = columns;
    auto result = numeric::linear(matrix, weights,
        bias ? std::optional{detail::data_of(*bias)} : std::nullopt, std::move(output_shape));
    std::vector edges{detail::gradient_edge(input), detail::gradient_edge(weight),
        bias ? detail::gradient_edge(*bias) : detail::Edge{}};
    std::optional<detail::SavedTensor> saved_input = edges[1].node ? std::optional{matrix} : std::nullopt;
    std::optional<detail::SavedTensor> saved_weight = edges[0].node ? std::optional{weights} : std::nullopt;
    return detail::record(std::move(result), std::move(edges),
        [saved_input = std::move(saved_input), saved_weight = std::move(saved_weight),
         input_shape = input.shape(), weight_shape = weight.shape(), bias_shape = bias ? bias->shape() : Shape{},
         rows, columns, compute_dtype](const auto& grad, auto& sink, const auto& inputs) {
            const bool input_gradient = sink.needs(inputs[0]), weight_gradient = sink.needs(inputs[1]);
            if (input_gradient || weight_gradient) {
                const auto seed = numeric::reshape(numeric::cast(grad, compute_dtype), {rows, columns});
                // dX = dY W，dW = dY^T X；合并行轴后直接写入两个最终累计目标。
                if (input_gradient) matmul_gradient(inputs[0], seed, *saved_weight, input_shape, sink);
                if (weight_gradient) {
                    matmul_gradient(inputs[1], numeric::transpose(seed, 0, 1), *saved_input, weight_shape, sink);
                }
            }
            // 偏置沿前导维求和，使用原始累计精度的 seed。
            if (sink.needs(inputs[2])) sink.add_reduced(inputs[2], grad, bias_shape);
        });
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    const bool left_vector = a.ndim() == 1;
    const bool right_vector = b.ndim() == 1;
    if (left_vector || right_vector) {
        // 向量补成行矩阵或列矩阵，批量广播与 VJP 继续沿矩阵路径执行。
        auto left = left_vector ? a.reshape({1, a.size(0)}) : a;
        auto right = right_vector ? b.reshape({b.size(0), 1}) : b;
        auto result = matmul(left, right);
        auto shape = result.shape();
        if (left_vector) {
            shape.erase(shape.end() - 2);
        }
        if (right_vector) {
            shape.pop_back();
        }
        return result.reshape(std::move(shape));
    }

    const auto compute_dtype = matrix_compute_dtype(a);
    auto left = detail::numeric::cast(detail::data_of(a), compute_dtype);
    auto right = detail::numeric::cast(detail::data_of(b), compute_dtype);
    std::vector edges{detail::gradient_edge(a), detail::gradient_edge(b)};
    std::optional<detail::SavedTensor> saved_a = edges[1].node ? std::optional{left} : std::nullopt;
    std::optional<detail::SavedTensor> saved_b = edges[0].node ? std::optional{right} : std::nullopt;
    return detail::record(detail::numeric::matmul(left, right), std::move(edges),
        [saved_a = std::move(saved_a), saved_b = std::move(saved_b), a_shape = a.shape(), b_shape = b.shape(), compute_dtype]
        (const auto& grad, auto& sink, const auto& inputs) {
            namespace numeric = detail::numeric;
            const auto seed = numeric::cast(grad, compute_dtype);
            // dA = dC B^T，dB = A^T dC；共享批维再按原输入形状归约。
            if (sink.needs(inputs[0])) {
                const auto rank = static_cast<Index>(saved_b->layout.sizes.size());
                matmul_gradient(inputs[0], seed, numeric::transpose(*saved_b, rank - 2, rank - 1), a_shape, sink);
            }
            if (sink.needs(inputs[1])) {
                const auto rank = static_cast<Index>(saved_a->layout.sizes.size());
                matmul_gradient(inputs[1], numeric::transpose(*saved_a, rank - 2, rank - 1), seed, b_shape, sink);
            }
        });
}

} // namespace dl
