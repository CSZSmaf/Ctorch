#include "backends/matrix.h"
#include "core/dispatch.h"
#include "core/iteration.h"
#include "kernels/cpu/matmul.h"
#include "kernels/cpu/ops.h"

#ifdef CTORCH_USE_BLAS
#include "backends/blas.h"
#endif

namespace dl::detail::cpu {

void matmul_into(TensorData& result, const TensorData& left, const TensorData& right, bool accumulate) {
    if (result.layout.numel() == 0) return;
    const auto left_rank = left.layout.sizes.size();
    const auto right_rank = right.layout.sizes.size();
    const auto rows = left.layout.sizes[left_rank - 2];
    const auto inner = left.layout.sizes[left_rank - 1];
    const auto columns = right.layout.sizes[right_rank - 1];
    if (inner == 0) {
        // 空内维的乘积为零；首次写入产生零值，累计时保留既有贡献。
        if (!accumulate) visit_floating(result.dtype, [&]<class T> {
            std::fill_n(result.mutable_data<T>(), result.layout.numel(), T{0});
        });
        return;
    }
    // CPU 参考路径展开低精度输入，FP32 结果仍直接写入调用方的目标。
    if (is_low_precision(left.dtype)) {
        cpu::matmul_into(result, cpu::cast(left, DType::f32), cpu::cast(right, DType::f32), accumulate);
        return;
    }
    const Shape batch_shape(result.layout.sizes.begin(), result.layout.sizes.end() - 2);

    auto a = left;
    auto b = right;
#ifdef CTORCH_USE_BLAS
    // 能由 transpose 与 leading dimension 描述的布局直接交给 BLAS，其余布局按逻辑顺序打包。
    if (!blas::matrix_layout(a)) {
        a = clone(a);
    }
    if (!blas::matrix_layout(b)) {
        b = clone(b);
    }
#endif
    IterationPlan<2> plan(batch_shape, {broadcast_strides(matrix_batches(a), batch_shape),
                                       broadcast_strides(matrix_batches(b), batch_shape)});
    plan.for_each([&](Index batch, const auto& offsets) {
#ifdef CTORCH_USE_BLAS
        blas::matmul(a, offsets[0], b, offsets[1], result, batch * rows * columns,
                     rows, columns, inner, accumulate);
#else
        visit_floating(left.dtype, [&]<class T> {
            cpu::matmul(rows, columns, inner, a.data<T>() + offsets[0],
                        a.layout.strides[left_rank - 2], a.layout.strides[left_rank - 1],
                        b.data<T>() + offsets[1], b.layout.strides[right_rank - 2],
                        b.layout.strides[right_rank - 1], result.mutable_data<T>() + batch * rows * columns,
                        accumulate);
        });
#endif
    });
}

} // namespace dl::detail::cpu
