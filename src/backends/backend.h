#pragma once

#include "ops/numeric.h"

namespace dl::detail {

/// 库卷积按一次前向或一种 VJP 执行；三个方向都显式给出目标形状。
/// 两个操作数依次为前向的 X/W、输入梯度的 dY/W、权重梯度的 dY/X。
enum class ConvolutionPass { forward, input_gradient, weight_gradient };
using ConvolutionKernel = TensorData (*)(ConvolutionPass, const TensorData&, const TensorData&,
                                         Shape, const Conv2dOptions&);

/// 准备入口统一精度和库布局；执行入口接收已准备的数据，反向可复用前向保存值。
struct ConvolutionBackend {
    TensorData (*prepare)(const TensorData&, DType);
    ConvolutionKernel execute;
};

/// 每个后端提供同一组无图数值操作，张量视图与 VJP 共用上层布局规则。
/// 函数表在构建时确定，设备选择发生在一次算子的入口。
struct Backend {
    Allocation (*allocate)(std::size_t, Device);
    void (*synchronize)(Device);
    void (*transfer)(void*, Device, const void*, Device, std::size_t);

    TensorData (*random)(Shape, numeric::Random, double, double, TensorOptions, Generator&);
    numeric::DropoutResult (*dropout)(const TensorData&, double, Generator&);
    TensorData (*random_keyed)(Shape, numeric::Random, double, double, TensorOptions, RandomKey);
    numeric::DropoutResult (*dropout_keyed)(const TensorData&, double, RandomKey);
    TensorData (*masked_scale)(const TensorData&, const TensorData&, double);

    TensorData (*full)(Shape, double, DType, Device);
    TensorData (*clone)(const TensorData&);
    TensorData (*cast)(const TensorData&, DType);
    TensorData (*cast_add)(const TensorData&, const TensorData&);
    std::array<TensorData, 2> (*cast_pair)(const TensorData&, const TensorData&, DType);
    void (*copy_into)(TensorData&, const TensorData&);
    void (*axpy)(TensorData&, const TensorData&, double);
    TensorData (*unary)(const TensorData&, numeric::Unary);
    void (*unary_backward_into)(TensorData&, const TensorData&, const TensorData&, numeric::Unary, bool);
    TensorData (*add)(const TensorData&, const TensorData&);
    TensorData (*add_relu)(const TensorData&, const TensorData&);
    TensorData (*sub)(const TensorData&, const TensorData&);
    TensorData (*mul)(const TensorData&, const TensorData&);
    TensorData (*div)(const TensorData&, const TensorData&);
    TensorData (*pow)(const TensorData&, const TensorData&);
    void (*scale_into)(TensorData&, const TensorData&, double, bool);
    TensorData (*where)(const TensorData&, const TensorData&, const TensorData&);
    TensorData (*compare)(const TensorData&, const TensorData&, numeric::Compare);
    TensorData (*sum)(const TensorData&, const Shape&, KeepDim);
    // 统计内核可在写出前完成宽精度累积；其他后端复用数值层的组合实现。
    TensorData (*mean)(const TensorData&, const Shape&, KeepDim) = nullptr;
    numeric::VarianceResult (*variance)(const TensorData&, const Shape&, double, KeepDim) = nullptr;
    void (*sum_into)(TensorData&, const TensorData&, const Shape&, bool);
    TensorData (*amax)(const TensorData&, const Shape&, KeepDim);
    TensorData (*arg_extreme)(const TensorData&, Index, KeepDim, bool);
    TensorData (*cumsum)(const TensorData&, Index, bool);
    TensorData (*argsort)(const TensorData&, Index, bool);
    void (*matmul_into)(TensorData&, const TensorData&, const TensorData&, bool);
    // 可选的矩阵乘加偏置实现；返回 false 时由数值层组合原有操作。
    bool (*linear_into)(TensorData&, const TensorData&, const TensorData&, const TensorData&) = nullptr;
    TensorData (*softmax)(const TensorData&, Index, bool);
    TensorData (*softmax_backward)(const TensorData&, const TensorData&, Index, bool);
    void (*normalization)(const TensorData&, const std::optional<TensorData>&,
        const std::optional<TensorData>&, Index, double, numeric::NormalizationKind,
        numeric::NormalizationResult&);
    void (*normalization_backward)(const TensorData&, const TensorData&, const TensorData&,
        const std::optional<TensorData>&, Index, numeric::NormalizationKind,
        numeric::NormalizationGradients&);
    void (*batch_normalization)(const TensorData&, const std::optional<TensorData>&,
        const std::optional<TensorData>&, std::optional<numeric::BatchNormalizationState>&,
        bool, double, double, numeric::NormalizationResult&);
    void (*batch_normalization_backward)(const TensorData&, const TensorData&, const TensorData&,
        const std::optional<TensorData>&, bool, numeric::NormalizationGradients&);
    TensorData (*gather)(const TensorData&, Index, const TensorData&);
    TensorData (*repeat)(const TensorData&, Shape);
    void (*repeat_backward_into)(TensorData&, const TensorData&);
    void (*scatter_add_into)(TensorData&, Index, const TensorData&, const TensorData&);
    void (*cross_entropy)(const TensorData&, const TensorData&, LossReduction, std::optional<Index>,
        numeric::CrossEntropyResult&);
    TensorData (*binary_logits)(const TensorData&, const TensorData&);
    void (*binary_logits_backward_into)(TensorData&, const TensorData&, const TensorData&, const TensorData&,
        double, bool, bool);
    void (*cross_entropy_backward_into)(TensorData&, const TensorData&, const TensorData&, const TensorData&,
        const TensorData&, LossReduction, std::optional<Index>, bool);
    void (*momentum)(TensorData&, const TensorData&, double, double);
    TensorData (*unscale_gradients)(std::span<TensorData>, double);
    TensorData (*clip_grad_norm)(std::span<TensorData>, double, double);
    void (*clip_grad_value)(std::span<TensorData>, double);
    void (*sgd)(std::span<const numeric::SGDUpdate>);
    void (*adam)(std::span<const numeric::AdamUpdate>);
    void (*rmsprop)(std::span<const numeric::RMSpropUpdate>);
    void (*adagrad)(std::span<const numeric::AdagradUpdate>);

    TensorData (*im2col)(const TensorData&, Size2, const Conv2dOptions&);
    void (*col2im_into)(TensorData&, const TensorData&, Size2, const Conv2dOptions&, bool);
    numeric::MaxPoolResult (*max_pool2d)(const TensorData&, const Pool2dOptions&, bool);
    void (*max_pool2d_backward_into)(TensorData&, const TensorData&, const TensorData&);
    TensorData (*avg_pool2d)(const TensorData&, const Pool2dOptions&, bool);
    void (*avg_pool2d_backward_into)(TensorData&, const TensorData&, const Pool2dOptions&, bool);
    TensorData (*adaptive_avg_pool2d)(const TensorData&, Size2);
    void (*adaptive_avg_pool2d_backward_into)(TensorData&, const TensorData&);
    TensorData (*interpolate)(const TensorData&, Size2, InterpolationMode, bool);
    void (*interpolate_backward_into)(TensorData&, const TensorData&, InterpolationMode, bool);

    // 构建配置选择库卷积准备与执行入口；空 execute 表示使用共用的 im2col/GEMM 组合。
    ConvolutionBackend convolution;
};

const Backend& backend(Device device);
const Backend& cpu_backend();
const Backend& cuda_backend();

} // namespace dl::detail
