#pragma once

#include "dl/onnx_export.h"
#include "dl/plan.h"

#include <filesystem>

namespace dl::nn { class Embedding; class TransformerDecoder; class Linear; }

namespace dl::onnx {

struct Binding { plan::Value input; Tensor value; };

/// 将固定形状的 FP32/FP64 推理图保存为 ONNX opset 18。绑定的输入成为常量权重，
/// 其余输入保留 Graph 中的名称；输出依次命名为 output_0、output_1 等。
/// 支持常见线性、激活、卷积、池化与归一化描述，未使用的纯计算自动移除。
/// 无描述的回调、状态更新和训练随机操作在导出时报告对应节点。
DL_ONNX_API void save(const std::filesystem::path& path, const plan::Graph& graph,
    std::span<const plan::Value> outputs, std::span<const Binding> weights = {});
inline void save(const std::filesystem::path& path, const plan::Graph& graph,
    std::initializer_list<plan::Value> outputs, std::initializer_list<Binding> weights = {}) {
    save(path, graph, std::span(outputs.begin(), outputs.size()), std::span(weights.begin(), weights.size()));
}

/// 导出 FP32 解码器推理图。输入特征为 [batch,tokens,width]，输出同形。
/// 每层接收 past_key_i、past_value_i 并返回 present_key_i、present_value_i。
/// 首次调用传入长度为零的缓存。批量、新块长度和历史长度均在运行时确定。
DL_ONNX_API void save_decoder(const std::filesystem::path& path, const nn::TransformerDecoder& decoder);
/// 带词嵌入和输出投影的模型直接接收 int64 input_ids，并输出词表 logits。
DL_ONNX_API void save_decoder(const std::filesystem::path& path, const nn::Embedding& embedding,
                              const nn::TransformerDecoder& decoder, const nn::Linear& output);

} // namespace dl::onnx
