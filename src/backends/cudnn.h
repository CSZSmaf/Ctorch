#pragma once

#include "backends/backend.h"
#include "backends/cuda_context.h"

#include <array>

namespace dl::detail::cuda {

/// cuDNN 的 handle、图计划和可复用工作区归属设备上下文，每次执行借用当前操作数。
std::shared_ptr<DnnContext> make_dnn_context(Device device, cudaStream_t stream, cudaMemPool_t pool);
/// 工作区沿计算 stream 归还；调用方在需要回收池内存时再等待该 stream。
void release_dnn_workspace(DnnContext& context);
/// 按目标精度一次完成转换和打包，返回逻辑 NCHW / KCRS 的数据描述。
TensorData prepare_convolution(const TensorData& input, DType dtype);
/// 两个输入已按相同精度准备，结果保留库精度和对应物理布局。
TensorData convolution(ConvolutionPass pass, const TensorData& first, const TensorData& second,
                       Shape output_shape, const Conv2dOptions& options);

struct AttentionResult {
    TensorData output;
    std::optional<TensorData> statistics;
};

/// Q/K/V 与可选加性 mask 为连续的低精度四维数据，统计量使用 FP32。
AttentionResult attention_forward(const std::array<TensorData, 3>& inputs,
    const std::optional<TensorData>& bias, AttentionComputeOptions options, RandomKey random, bool save_statistics);
/// 输出与种子使用前向低精度；库计算三个方向，外层累计器再选择所需的梯度。
std::array<TensorData, 3> attention_backward(const std::array<TensorData, 3>& inputs,
    const TensorData& output, const TensorData& statistics, const TensorData& grad,
    const std::optional<TensorData>& bias, AttentionComputeOptions options, RandomKey random);

} // namespace dl::detail::cuda
