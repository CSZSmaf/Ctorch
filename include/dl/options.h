#pragma once

#include "dl/types.h"

namespace dl {

/// 损失保留逐项结果，或将全部损失项求均值、求和。
enum class LossReduction { none, mean, sum };

/// 一个整数描述两个相等的空间维度，两个整数分别描述高度与宽度。
struct Size2 {
    Index height;
    Index width;
    constexpr Size2(Index both = 1) : height(both), width(both) {}
    constexpr Size2(Index height, Index width) : height(height), width(width) {}
};

struct Conv2dOptions {
    Size2 stride{1};
    Size2 padding{0};
    Size2 dilation{1};
    Index groups = 1;
};

struct Conv1dOptions {
    Index stride = 1;
    Index padding = 0;
    Index dilation = 1;
    Index groups = 1;
};

struct Pool1dOptions {
    Index kernel = 2;
    Index stride = 2;
    Index padding = 0;
};

/// nearest 使用左端比例坐标，nearest_exact 使用像素中心，bilinear 使用四邻域线性权重。
enum class InterpolationMode { nearest, nearest_exact, bilinear };

/// kernel 与 stride 为正，padding 非负；输出尺寸向下取整，每个窗口与输入至少有一个交点。
struct Pool2dOptions {
    Size2 kernel{2};
    Size2 stride{2};
    Size2 padding{0};
};

struct BatchNormOptions {
    double epsilon = 1e-5;
    double momentum = 0.1;
    bool affine = true;
    bool track_running_stats = true;
};

/// math 组合普通算子；cudnn 显式选择 CUDA FP16/BF16 的融合注意力。
enum class AttentionBackend { math, cudnn };

struct AttentionComputeOptions {
    double dropout = 0.0;
    bool causal = false;
    AttentionBackend backend = AttentionBackend::math;
};

struct AttentionOptions {
    double dropout = 0.0;
    bool bias = true;
};

enum class RNNActivation { tanh, relu };
enum class RecurrentKind { rnn_tanh, rnn_relu, gru, lstm };

/// dropout 仅作用于相邻循环层之间；batch_first 不改变隐藏状态的轴顺序。
struct RecurrentOptions {
    Index layers = 1;
    bool bias = true;
    bool batch_first = false;
    bool bidirectional = false;
    double dropout = 0.0;
};

} // namespace dl
