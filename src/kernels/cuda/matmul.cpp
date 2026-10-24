#include "kernels/cuda/ops.h"

#include "backends/cuda_context.h"
#include "backends/matrix.h"
#include "core/iteration.h"

namespace dl::detail::cuda {

void matmul_into(TensorData& result, const TensorData& left, const TensorData& right, bool accumulate) {
    if (result.layout.numel() == 0) return;
    const auto m = left.layout.sizes[left.layout.sizes.size() - 2];
    const auto k = left.layout.sizes.back();
    const auto n = right.layout.sizes.back();
    if (k == 0) {
        // 空内维不提交 GEMM，覆盖模式在同一计算 stream 写入零乘积。
        if (!accumulate) {
            Execution execution(left);
            check(cudaMemsetAsync(result.mutable_address(), 0,
                static_cast<std::size_t>(result.layout.numel()) * element_size(result.dtype), execution.stream()));
        }
        return;
    }
    Shape batches(result.layout.sizes.begin(), result.layout.sizes.end() - 2);
    auto a = matrix_format(left) ? left : clone(left);
    auto b = matrix_format(right) ? right : clone(right);
    const auto a_format = *matrix_format(a);
    const auto b_format = *matrix_format(b);
    Execution execution(left);
    auto a_strides = broadcast_strides(matrix_batches(a), batches);
    auto b_strides = broadcast_strides(matrix_batches(b), batches);
    Index batch_count = 1, a_step = 0, b_step = 0;
    // 从最内层批维合并等间距矩阵。注意力的 head 维可单独成组，外层 batch 继续遍历。
    // 零 stride 的广播批维留在外层，使每组满足 cuBLAS 的非零矩阵间距约定。
    for (std::size_t dim = batches.size(); dim-- > 0;) {
        if (batches[dim] != 1) {
            if (batch_count == 1) {
                if (a_strides[dim] == 0 || b_strides[dim] == 0) break;
                a_step = a_strides[dim];
                b_step = b_strides[dim];
            } else if (a_strides[dim] != a_step * batch_count || b_strides[dim] != b_step * batch_count) {
                break;
            }
            batch_count *= batches[dim];
        }
        batches.pop_back();
        a_strides.pop_back();
        b_strides.pop_back();
    }
    IterationPlan<2> plan(std::move(batches), {std::move(a_strides), std::move(b_strides)});
    const auto a_transpose = a_format.transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto b_transpose = b_format.transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float alpha = 1.0f;
    // beta=0 覆盖未初始化的目标，beta=1 将本次乘积并入已有梯度。
    const float beta = accumulate ? 1.0f : 0.0f;
    plan.for_each([&](Index batch, const auto& offsets) {
        // cuBLAS 使用列主序；C^T = B^T A^T 使行主序结果直接写入原输出缓冲。
        auto* output = result.mutable_data<float>() + batch * batch_count * m * n;
        if (is_low_precision(left.dtype)) {
            const auto dtype = left.dtype == DType::f16 ? CUDA_R_16F : CUDA_R_16BF;
            const auto* a_pointer = static_cast<const std::byte*>(a.address()) + offsets[0] * element_size(a.dtype);
            const auto* b_pointer = static_cast<const std::byte*>(b.address()) + offsets[1] * element_size(b.dtype);
            // 16 位输入参与乘法，累积和输出均为 FP32；批次规划与普通 GEMM 共用。
            if (batch_count == 1) {
                check(cublasGemmEx(execution.context.blas.get(), b_transpose, a_transpose,
                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                    b_pointer, dtype, static_cast<int>(b_format.leading_dimension),
                    a_pointer, dtype, static_cast<int>(a_format.leading_dimension), &beta,
                    output, CUDA_R_32F, static_cast<int>(n), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                check(cublasGemmStridedBatchedEx(execution.context.blas.get(), b_transpose, a_transpose,
                    static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                    b_pointer, dtype, static_cast<int>(b_format.leading_dimension), b_step,
                    a_pointer, dtype, static_cast<int>(a_format.leading_dimension), a_step, &beta,
                    output, CUDA_R_32F, static_cast<int>(n), m * n, static_cast<int>(batch_count),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
            return;
        }
        if (batch_count == 1) {
            check(cublasSgemm(execution.context.blas.get(), b_transpose, a_transpose,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                b.data<float>() + offsets[1], static_cast<int>(b_format.leading_dimension),
                a.data<float>() + offsets[0], static_cast<int>(a_format.leading_dimension), &beta,
                output, static_cast<int>(n)));
        } else {
            check(cublasSgemmStridedBatched(execution.context.blas.get(), b_transpose, a_transpose,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                b.data<float>() + offsets[1], static_cast<int>(b_format.leading_dimension), b_step,
                a.data<float>() + offsets[0], static_cast<int>(a_format.leading_dimension), a_step, &beta,
                output, static_cast<int>(n), m * n, static_cast<int>(batch_count)));
        }
    });
}

} // namespace dl::detail::cuda
