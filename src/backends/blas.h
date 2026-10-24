#pragma once

#include "core/tensor_data.h"

namespace dl::detail::blas {

bool matrix_layout(const TensorData& input);
void matmul(const TensorData& a, Index a_offset, const TensorData& b, Index b_offset,
            TensorData& output, Index output_offset, Index rows, Index columns, Index inner,
            bool accumulate = false);

} // namespace dl::detail::blas
