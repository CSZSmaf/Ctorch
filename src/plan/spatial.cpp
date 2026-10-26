#include "dl/plan.h"

#include "core/resampling.h"
#include "core/spatial.h"
#include "dl/ops.h"

namespace dl::plan::ops {
namespace {

OutputSpec spatial_spec(Shape shape, const TensorSpec& input) {
    return {contiguous_spec(std::move(shape), accumulation_dtype(input.dtype), input.device)};
}

// cuDNN 可返回通道连续布局；计划统一交付规范连续步长，单位轴只需调整视图。
Tensor convolution_result(Tensor result) {
    if (result.strides() == detail::contiguous_strides(result.shape())) return result;
    return result.is_contiguous() ? result.reshape(result.shape()) : result.clone();
}

Shape dimensions(Index value) { return {value}; }
Shape dimensions(Size2 value) { return {value.height, value.width}; }
template <class Options> std::vector<Attribute> window_attributes(Options options) {
    auto padding = dimensions(options.padding);
    const auto original = padding;
    padding.insert(padding.end(), original.begin(), original.end());
    return {{"strides", dimensions(options.stride)}, {"pads", std::move(padding)}};
}
template <class Options> OperatorInfo convolution_info(Options options) {
    auto attributes = window_attributes(options);
    attributes.emplace_back("dilations", dimensions(options.dilation));
    attributes.emplace_back("group", options.groups);
    return {"conv", std::move(attributes)};
}
template <class Options> Operation pool_info(Operation operation, Options options, bool average, bool include_pad = true) {
    auto attributes = window_attributes(options);
    attributes.emplace_back("kernel_shape", dimensions(options.kernel));
    if (average) attributes.emplace_back("count_include_pad", Index(include_pad));
    operation.description = {average ? "avg_pool" : "max_pool", std::move(attributes)};
    return operation;
}

Shape spatial_shape(Shape shape, Pool1dOptions options) {
    shape.back() = detail::spatial_extent(shape.back(), options.kernel, options.stride, options.padding);
    return shape;
}

Shape spatial_shape(const Shape& shape, Pool2dOptions options) {
    return detail::spatial_output(detail::spatial_geometry(shape, options.kernel, options.stride, options.padding));
}

Shape spatial_shape(Shape shape, Index output_size) {
    shape.back() = output_size;
    return shape;
}

Shape spatial_shape(Shape shape, Size2 output_size) {
    return detail::resampling_shape(std::move(shape), output_size);
}

/// 池化与插值按各自规则推导空间尺寸，共用单输入、独立结果的描述。
template <class Options, class Forward>
Operation spatial_operation(std::string name, Options options, Forward forward) {
    return {std::move(name), [options](std::span<const TensorSpec> inputs) {
        return spatial_spec(spatial_shape(inputs[0].shape, options), inputs[0]);
    }, [options, forward](std::span<const Tensor> inputs) {
        return forward(inputs[0], options);
    }};
}

} // namespace

Operation conv1d(Conv1dOptions options) {
    return {"conv1d", [options](std::span<const TensorSpec> inputs) {
        auto shape = inputs[0].shape;
        const auto& weight = inputs[1].shape;
        shape[shape.size() - 2] = weight[0];
        shape.back() = detail::spatial_extent(shape.back(), weight[2],
            options.stride, options.padding, options.dilation);
        return spatial_spec(std::move(shape), inputs[0]);
    }, [options](std::span<const Tensor> inputs) {
        const auto bias = inputs.size() == 3 ? std::optional{inputs[2]} : std::nullopt;
        return convolution_result(dl::conv1d(inputs[0], inputs[1], bias, options));
    }, Effect::none, Semantic::custom, convolution_info(options)};
}

Operation conv2d(Conv2dOptions options) {
    return {"conv2d", [options](std::span<const TensorSpec> inputs) {
        const auto& input = inputs[0];
        const auto& weight = inputs[1].shape;
        auto shape = detail::spatial_output(detail::spatial_geometry(input.shape,
            {weight[2], weight[3]}, options.stride, options.padding, options.dilation));
        shape[1] = weight[0];
        return spatial_spec(std::move(shape), input);
    }, [options](std::span<const Tensor> inputs) {
        const auto bias = inputs.size() == 3 ? std::optional{inputs[2]} : std::nullopt;
        return convolution_result(dl::conv2d(inputs[0], inputs[1], bias, options));
    }, Effect::none, Semantic::custom, convolution_info(options)};
}

Operation max_pool1d(Pool1dOptions options) { return pool_info(spatial_operation("max_pool1d", options, dl::max_pool1d), options, false); }

Operation avg_pool1d(Pool1dOptions options, bool count_include_pad) {
    return pool_info(spatial_operation("avg_pool1d", options, [count_include_pad](const Tensor& input, Pool1dOptions window) {
        return dl::avg_pool1d(input, window, count_include_pad);
    }), options, true, count_include_pad);
}

Operation max_pool2d(Pool2dOptions options) { return pool_info(spatial_operation("max_pool2d", options, dl::max_pool2d), options, false); }

Operation avg_pool2d(Pool2dOptions options, bool count_include_pad) {
    return pool_info(spatial_operation("avg_pool2d", options, [count_include_pad](const Tensor& input, Pool2dOptions window) {
        return dl::avg_pool2d(input, window, count_include_pad);
    }), options, true, count_include_pad);
}

Operation adaptive_avg_pool1d(Index output_size) {
    auto operation = spatial_operation("adaptive_avg_pool1d", output_size, dl::adaptive_avg_pool1d);
    if (output_size == 1) operation.description = {"global_avg_pool", {{"spatial_rank", Index{1}}}};
    return operation;
}

Operation adaptive_avg_pool2d(Size2 output_size) {
    auto operation = spatial_operation("adaptive_avg_pool2d", output_size, dl::adaptive_avg_pool2d);
    if (output_size.height == 1 && output_size.width == 1) operation.description = {"global_avg_pool", {{"spatial_rank", Index{2}}}};
    return operation;
}

Operation interpolate(Size2 output_size, InterpolationMode mode, bool align_corners) {
    return spatial_operation("interpolate", output_size, [mode, align_corners](const Tensor& input, Size2 size) {
        return dl::interpolate(input, size, mode, align_corners);
    });
}

} // namespace dl::plan::ops
