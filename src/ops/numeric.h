#pragma once

#include "core/tensor_data.h"
#include "dl/options.h"

#include <array>

namespace dl { class Generator; struct RandomKey; }

namespace dl::detail::numeric {

enum class Unary { exp, log, sqrt, tanh, sigmoid, relu, gelu, log1p, expm1, abs, sin, cos, softplus,
                   tan, asin, acos, atan, silu };
enum class Compare { eq, ne, lt, le, gt, ge };
enum class Random { uniform, normal, bernoulli };

struct DropoutResult { TensorData output, mask; };
TensorData random(Shape shape, Random distribution, double first, double second,
                  TensorOptions options, Generator& generator);
DropoutResult dropout(const TensorData& input, double probability, Generator& generator);
TensorData random(Shape shape, Random distribution, double first, double second,
                  TensorOptions options, RandomKey key);
DropoutResult dropout(const TensorData& input, double probability, RandomKey key);
TensorData masked_scale(const TensorData& input, const TensorData& mask, double scale);

TensorData full(Shape shape, double value, DType dtype, Device device = Device::cpu());
TensorData to(const TensorData& input, Device device);
TensorData clone(const TensorData& input);
TensorData cast(const TensorData& input, DType dtype);
/// 低精度输入转成 FP32 后加上可广播的 FP32 偏置，共用一次结果遍历。
TensorData cast_add(const TensorData& input, const TensorData& offset);
std::array<TensorData, 2> cast_pair(const TensorData& first, const TensorData& second, DType dtype);
void copy_into(TensorData& target, const TensorData& source);
void add_into(TensorData& target, const TensorData& contribution);
void axpy(TensorData& target, const TensorData& input, double alpha);

struct SGDStep {
    double learning_rate, momentum, dampening, weight_decay;
    bool nesterov, initialize_velocity;
};

struct SGDUpdate {
    TensorData* parameter;
    const TensorData* gradient;
    TensorData* velocity;
    SGDStep step;
};

struct AdamStep {
    double learning_rate, beta1, beta2, epsilon, weight_decay;
    double inverse_bias1, inverse_bias2;
    bool decoupled = true;
};

/// 描述符借用本次调用中存活的数据对象，参数与两个矩缓冲分别拥有独立写入区间。
struct AdamUpdate {
    TensorData* parameter;
    const TensorData* gradient;
    TensorData* first;
    TensorData* second;
    AdamStep step;
};

struct RMSpropStep {
    double learning_rate, alpha, epsilon, weight_decay, momentum;
};

/// 可选指针表示本步参与递推的动量和梯度均值；其余状态由调用方保留。
struct RMSpropUpdate {
    TensorData* parameter;
    const TensorData* gradient;
    TensorData* square_average;
    TensorData* momentum;
    TensorData* gradient_average;
    RMSpropStep step;
};

struct AdagradStep {
    double learning_rate, epsilon, weight_decay;
};

struct AdagradUpdate {
    TensorData* parameter;
    const TensorData* gradient;
    TensorData* sum;
    AdagradStep step;
};

void momentum(TensorData& buffer, const TensorData& gradient, double decay, double scale);
/// 非空列表中的同设备连续梯度原地乘倒数，返回设备 f32 标量，非有限值存在时为一。
TensorData unscale_gradients(std::span<TensorData> gradients, double inverse_scale);
/// 同设备同类型的非空列表直接累计全局范数，再原地缩放，返回裁剪前的设备标量。
TensorData clip_grad_norm(std::span<TensorData> gradients, double max_norm, double order);
/// 逐值裁剪按设备分段，CPU 可包含不同浮点类型。
void clip_grad_value(std::span<TensorData> gradients, double bound);
/// 列表按设备分段，CPU 沿各自布局更新，CUDA 合并连续参数的启动。
void sgd(std::span<const SGDUpdate> updates);
void adam(std::span<const AdamUpdate> updates);
void rmsprop(std::span<const RMSpropUpdate> updates);
/// learning_rate 已包含按本参数更新次数计算的衰减。
void adagrad(std::span<const AdagradUpdate> updates);

TensorData unary(const TensorData& input, Unary operation);
/// grad 与 saved 同形同设备，grad 使用 saved 的累积类型，按各自布局直接计算 VJP。
TensorData unary_backward(const TensorData& grad, const TensorData& saved, Unary operation);
/// 写回接口接收独立连续目标，accumulate 选择覆盖或累加；保存值与梯度保留各自布局。
void unary_backward_into(TensorData& target, const TensorData& grad, const TensorData& saved,
                         Unary operation, bool accumulate);
TensorData add(const TensorData& a, const TensorData& b);
TensorData add_relu(const TensorData& a, const TensorData& b);
TensorData sub(const TensorData& a, const TensorData& b);
TensorData mul(const TensorData& a, const TensorData& b);
TensorData div(const TensorData& a, const TensorData& b);
TensorData pow(const TensorData& a, const TensorData& b);
TensorData scale(const TensorData& input, double factor);
void scale_into(TensorData& target, const TensorData& input, double factor, bool accumulate);
TensorData where(const TensorData& mask, const TensorData& a, const TensorData& b);
TensorData compare(const TensorData& a, const TensorData& b, Compare operation);
TensorData concat(std::span<const TensorData> inputs, Index axis);

TensorData reshape(TensorData input, Shape shape);
TensorData transpose(const TensorData& input, Index first, Index second);
TensorData permute(const TensorData& input, const Shape& axes);
TensorData expand(const TensorData& input, Shape shape);
TensorData slice(const TensorData& input, Index axis, Index start, Index stop, Index step);
/// 切片以外的目标已具有初值，选中区域按模式覆盖或累计。
void slice_backward_into(TensorData& target, const TensorData& grad,
                         Index axis, Index start, Index step, bool accumulate);

Shape reduced_shape(const Shape& shape, const Shape& axes, KeepDim keep);
TensorData sum(const TensorData& input, const Shape& axes, KeepDim keep);
TensorData mean(const TensorData& input, const Shape& axes, KeepDim keep);
/// centered 保存减去均值后的值，供方差与标准差的 VJP 共用。
struct VarianceResult { TensorData output, centered; };
VarianceResult variance(const TensorData& input, const Shape& axes, double correction, KeepDim keep);
/// 目标连续且独立，按归约结果的逻辑顺序写入；目标形状可采用其连续重排。
void sum_into(TensorData& target, const TensorData& input, const Shape& axes, bool accumulate);
TensorData amax(const TensorData& input, const Shape& axes, KeepDim keep);
TensorData arg_extreme(const TensorData& input, Index axis, KeepDim keep, bool maximum);
TensorData cumsum(const TensorData& input, Index axis, bool reverse = false);
TensorData argsort(const TensorData& input, Index axis, bool descending);
TensorData sum_to_shape(const TensorData& input, Shape shape);
void sum_to_shape_into(TensorData& target, const TensorData& input, bool accumulate);
/// 两个操作数均已具有矩阵维；向量补维由公开 matmul 处理。
Shape matmul_shape(const TensorData& a, const TensorData& b);
TensorData matmul(const TensorData& a, const TensorData& b);
/// 目标连续、与输入存储独立，shape 为完整乘积，dtype 为累积精度。
void matmul_into(TensorData& target, const TensorData& a, const TensorData& b, bool accumulate);
/// 输入已合并为二维矩阵，output_shape 恢复前导维并决定偏置的广播形状。
TensorData linear(const TensorData& input, const TensorData& weight,
                  const std::optional<TensorData>& bias, Shape output_shape);
/// 两种概率表示共用稳定的行归约，低精度输入直接读取并以 FP32 计算。
TensorData softmax(const TensorData& input, Index axis, bool logarithmic = false);
TensorData softmax_backward(const TensorData& grad, const TensorData& output,
                           Index axis, bool logarithmic);

enum class NormalizationKind { layer, rms };
/// statistics 的两行分别存均值与逆标准差；CPU 用 f64 保存统计，CUDA 用 f32。
/// 逐行归一化在无需求导时省去该缓冲。
struct NormalizationResult { TensorData output; std::optional<TensorData> statistics; };
/// 三个连续目标依次对应输入、权重和偏置；累计标志决定覆盖还是追加贡献。
struct NormalizationGradients {
    std::optional<TensorData> input, weight, bias;
    std::array<bool, 3> accumulate{};
};
struct BatchNormalizationState { TensorData mean, variance; };
/// 输入保留存储精度，仿射参数使用累积类型；width 是末尾归一化维度的元素数。
NormalizationResult normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, Index width, double epsilon, NormalizationKind kind,
    bool save_statistics);
void normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    Index width, NormalizationKind kind, NormalizationGradients& targets);
/// 通道轴为 1；统计缓冲保存均值与逆标准差，运行方差使用无偏估计。
NormalizationResult batch_normalization(const TensorData& input, const std::optional<TensorData>& weight,
    const std::optional<TensorData>& bias, std::optional<BatchNormalizationState> state,
    bool training, double momentum, double epsilon);
void batch_normalization_backward(const TensorData& grad, const TensorData& input,
    const TensorData& statistics, const std::optional<TensorData>& weight,
    bool batch_statistics, NormalizationGradients& targets);

TensorData index_select(const TensorData& input, Index axis, const TensorData& indices);
/// 索引与周期反向只追加贡献，目标由调用方初始化。
void index_backward_into(TensorData& target, const TensorData& grad, Index axis, const TensorData& indices);
TensorData gather(const TensorData& input, Index axis, const TensorData& indices);
TensorData repeat(const TensorData& input, Shape shape);
void repeat_backward_into(TensorData& target, const TensorData& grad);
void scatter_add_into(TensorData& target, Index axis, const TensorData& indices, const TensorData& source);
TensorData scatter_add(const TensorData& input, Index axis, const TensorData& indices, const TensorData& source);

/// statistics 连续保存每行最大值与指数和；平均归约再使用末尾的有效项数，共 2*rows+1 个值。
/// 无需求导时省去统计缓冲；反向从原 logits 重算概率并直接写入累计目标。
struct CrossEntropyResult { TensorData loss; std::optional<TensorData> statistics; };
CrossEntropyResult cross_entropy(const TensorData& logits, const TensorData& labels,
    LossReduction reduction, std::optional<Index> ignore_index, bool save_statistics);
TensorData binary_logits(const TensorData& logits, const TensorData& labels);
void binary_logits_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, double factor, bool target_gradient, bool accumulate);
void cross_entropy_backward_into(TensorData& target, const TensorData& grad,
    const TensorData& logits, const TensorData& labels, const TensorData& statistics,
    LossReduction reduction, std::optional<Index> ignore_index, bool accumulate);

/// 窗口展开只复制存储值，保留输入 dtype；窗口重叠的反向汇总使用 GEMM 的累积类型。
TensorData im2col(const TensorData& input, Size2 kernel, const Conv2dOptions& options);
/// 将窗口贡献汇回连续图像目标，首次覆盖或在已有贡献上相加。
void col2im_into(TensorData& target, const TensorData& columns, Size2 kernel,
                 const Conv2dOptions& options, bool accumulate);

struct MaxPoolResult {
    TensorData values;
    std::optional<TensorData> indices;
};

/// 最大值位置只在需要反向时保存；两种反向入口向已有连续目标追加窗口贡献。
MaxPoolResult max_pool2d(const TensorData& input, const Pool2dOptions& options, bool save_indices);
void max_pool2d_backward_into(TensorData& target, const TensorData& grad, const TensorData& indices);
TensorData avg_pool2d(const TensorData& input, const Pool2dOptions& options, bool count_include_pad);
void avg_pool2d_backward_into(TensorData& target, const TensorData& grad, const Pool2dOptions& options,
                              bool count_include_pad);
TensorData adaptive_avg_pool2d(const TensorData& input, Size2 output_size);
/// 重采样读取最后两个空间轴，反向向连续目标追加区域或邻点贡献。
void adaptive_avg_pool2d_backward_into(TensorData& target, const TensorData& grad);
TensorData interpolate(const TensorData& input, Size2 output_size, InterpolationMode mode, bool align_corners);
void interpolate_backward_into(TensorData& target, const TensorData& grad, InterpolationMode mode, bool align_corners);

} // namespace dl::detail::numeric
