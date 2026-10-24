#pragma once

#include "dl/types.h"

#include <algorithm>

namespace dl::detail::cpu {

/// 参考 GEMM 直接使用行列步长；覆盖与累计共用转置、切片的数值循环。
template <class T>
void matmul(Index rows, Index columns, Index inner, const T* a, Index a_row, Index a_column,
            const T* b, Index b_row, Index b_column, T* output, bool accumulate = false) {
    if (!accumulate) std::fill_n(output, rows * columns, T{0});
    for (Index row = 0; row < rows; ++row) {
        for (Index k = 0; k < inner; ++k) {
            const T left = a[row * a_row + k * a_column];
            for (Index column = 0; column < columns; ++column) {
                output[row * columns + column] += left * b[k * b_row + column * b_column];
            }
        }
    }
}

} // namespace dl::detail::cpu
