#pragma once

#include "core/tensor_data.h"

#include <algorithm>
#include <optional>

namespace dl::detail {

struct MatrixFormat {
    bool transposed;
    Index leading_dimension;
};

/// 普通行主序与其转置由一个 leading dimension 表达，切片中的其他布局另行打包。
inline std::optional<MatrixFormat> matrix_format(const TensorData& input) {
    const auto rank = input.layout.sizes.size();
    const auto rows = input.layout.sizes[rank - 2];
    const auto columns = input.layout.sizes[rank - 1];
    const auto row_stride = input.layout.strides[rank - 2];
    const auto column_stride = input.layout.strides[rank - 1];
    if (column_stride == 1 && row_stride >= std::max(columns, Index{1})) {
        return MatrixFormat{false, row_stride};
    }
    if (row_stride == 1 && column_stride >= std::max(rows, Index{1})) {
        return MatrixFormat{true, column_stride};
    }
    return std::nullopt;
}

inline Layout matrix_batches(const TensorData& input) {
    return {Shape(input.layout.sizes.begin(), input.layout.sizes.end() - 2),
             Shape(input.layout.strides.begin(), input.layout.strides.end() - 2), 0};
}

} // namespace dl::detail
