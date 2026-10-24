#include "backends/blas.h"
#include "backends/matrix.h"

#include <cblas.h>

namespace dl::detail::blas {

bool matrix_layout(const TensorData& input) {
    return matrix_format(input).has_value();
}

void matmul(const TensorData& a, Index a_offset, const TensorData& b, Index b_offset,
            TensorData& output, Index output_offset, Index rows, Index columns, Index inner,
            bool accumulate) {
    const auto a_format = *matrix_format(a);
    const auto b_format = *matrix_format(b);
    const auto ta = a_format.transposed ? CblasTrans : CblasNoTrans;
    const auto tb = b_format.transposed ? CblasTrans : CblasNoTrans;
    const auto lda = static_cast<int>(a_format.leading_dimension);
    const auto ldb = static_cast<int>(b_format.leading_dimension);
    const auto m = static_cast<int>(rows);
    const auto n = static_cast<int>(columns);
    const auto k = static_cast<int>(inner);
    if (a.dtype == DType::f32) {
        cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, 1.0f,
                    a.data<float>() + a_offset, lda, b.data<float>() + b_offset, ldb,
                    accumulate ? 1.0f : 0.0f, output.mutable_data<float>() + output_offset, n);
    } else {
        cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, 1.0,
                    a.data<double>() + a_offset, lda, b.data<double>() + b_offset, ldb,
                    accumulate ? 1.0 : 0.0, output.mutable_data<double>() + output_offset, n);
    }
}

} // namespace dl::detail::blas
