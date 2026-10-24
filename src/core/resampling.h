#pragma once

#include "core/layout.h"
#include "dl/options.h"

#if defined(__CUDACC__)
#define CTORCH_RESAMPLE_HD __host__ __device__
#else
#define CTORCH_RESAMPLE_HD
#endif

namespace dl::detail {

struct AdaptiveWindow {
    Index plane, first_y, last_y, first_x, last_x;
    CTORCH_RESAMPLE_HD Index count() const { return (last_y - first_y) * (last_x - first_x); }
};

struct InterpolationPoint {
    Index indices[4]{};
    double weights[4]{};
    int count;
};

/// 空间几何使用逻辑坐标，数据的实际 stride 由 CPU Layout 或 CUDA IndexMap 处理。
struct ResamplingGeometry {
    Index height, width, output_height, output_width;

    CTORCH_RESAMPLE_HD Index offset(Index plane, Index y, Index x) const {
        return (plane * height + y) * width + x;
    }

    CTORCH_RESAMPLE_HD AdaptiveWindow window(Index index) const {
        const auto y = (index / output_width) % output_height;
        const auto x = index % output_width;
        return {index / (output_height * output_width),
            y * height / output_height, ((y + 1) * height + output_height - 1) / output_height,
            x * width / output_width, ((x + 1) * width + output_width - 1) / output_width};
    }

    CTORCH_RESAMPLE_HD static double coordinate(Index position, Index input, Index output, bool align_corners) {
        const double value = align_corners ? (output == 1 ? 0.0 : static_cast<double>(position) * (input - 1) / (output - 1))
            : (static_cast<double>(position) + 0.5) * input / output - 0.5;
        return value < 0.0 ? 0.0 : value > input - 1 ? static_cast<double>(input - 1) : value;
    }

    CTORCH_RESAMPLE_HD InterpolationPoint point(Index index, InterpolationMode mode, bool align_corners) const {
        const auto plane = index / (output_height * output_width);
        const auto y = (index / output_width) % output_height;
        const auto x = index % output_width;
        if (mode != InterpolationMode::bilinear) {
            const bool exact = mode == InterpolationMode::nearest_exact;
            const auto source_y = exact ? (2 * y + 1) * height / (2 * output_height) : y * height / output_height;
            const auto source_x = exact ? (2 * x + 1) * width / (2 * output_width) : x * width / output_width;
            return {{offset(plane, source_y, source_x)}, {1.0}, 1};
        }
        const double source_y = coordinate(y, height, output_height, align_corners);
        const double source_x = coordinate(x, width, output_width, align_corners);
        const auto y0 = static_cast<Index>(source_y), x0 = static_cast<Index>(source_x);
        const auto y1 = y0 + 1 < height ? y0 + 1 : y0;
        const auto x1 = x0 + 1 < width ? x0 + 1 : x0;
        const double wy = source_y - y0, wx = source_x - x0;
        return {{offset(plane, y0, x0), offset(plane, y0, x1), offset(plane, y1, x0), offset(plane, y1, x1)},
            {(1.0 - wy) * (1.0 - wx), (1.0 - wy) * wx, wy * (1.0 - wx), wy * wx}, 4};
    }
};

/// 前导维按图像平面展开，最后两个轴给出源与目标的空间尺寸。
inline ResamplingGeometry resampling_geometry(const Shape& input, const Shape& output) {
    return {input[input.size() - 2], input.back(), output[output.size() - 2], output.back()};
}

inline Shape resampling_shape(Shape shape, Size2 output) {
    shape[shape.size() - 2] = output.height;
    shape.back() = output.width;
    return shape;
}

} // namespace dl::detail

#undef CTORCH_RESAMPLE_HD
