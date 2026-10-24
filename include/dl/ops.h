#pragma once

#include "dl/tensor.h"
#include "dl/options.h"

#include <initializer_list>

namespace dl {

class DL_API Generator;
struct RandomKey;

/// CPU 数值计算使用 float32/float64，CUDA 浮点计算使用 float32。
/// 二元浮点运算按末尾维广播；所有操作数位于同一设备，浮点操作数使用同一 dtype。
/// bool 与 int64 分别用于选择掩码和索引，CUDA 内核接受至多 16 维的布局。
DL_API Tensor add(const Tensor& a, const Tensor& b);
/// 在同一次遍历中计算 relu(a + b)，按末尾维广播，保存输出用于一阶反向。
/// FP16/BF16 输入在内核中转为 FP32 计算并输出，CPU float64 保持双精度。
DL_API Tensor add_relu(const Tensor& a, const Tensor& b);
DL_API Tensor sub(const Tensor& a, const Tensor& b);
DL_API Tensor mul(const Tensor& a, const Tensor& b);
/// 两个输入的梯度分别为 g/b 和 -g*((a/b)/b)，广播轴按原输入形状求和。
DL_API Tensor div(const Tensor& a, const Tensor& b);
/// 实数幂按广播计算；负底数使用整数指数，指数梯度定义在正底数区域。
/// 零指数的底数梯度取零，零底数与正指数的指数梯度取零。
DL_API Tensor pow(const Tensor& base, const Tensor& exponent);
DL_API Tensor pow(const Tensor& base, double exponent);
DL_API Tensor pow(double base, const Tensor& exponent);
DL_API Tensor neg(const Tensor& input);
DL_API Tensor exp(const Tensor& input);
DL_API Tensor log(const Tensor& input);
/// log1p 和 expm1 分别直接计算 log(1+x) 与 exp(x)-1，保留零附近的小量精度。
DL_API Tensor log1p(const Tensor& input);
DL_API Tensor expm1(const Tensor& input);
/// abs 在零点的导数取零；sin、cos 使用弧度。
DL_API Tensor abs(const Tensor& input);
DL_API Tensor sin(const Tensor& input);
DL_API Tensor cos(const Tensor& input);
DL_API Tensor tan(const Tensor& input);
DL_API Tensor asin(const Tensor& input);
DL_API Tensor acos(const Tensor& input);
DL_API Tensor atan(const Tensor& input);
/// 将值限制到闭区间，minimum <= maximum；边界上的输入梯度取一。
DL_API Tensor clamp(const Tensor& input, double minimum, double maximum);
DL_API Tensor clamp_min(const Tensor& input, double minimum);
DL_API Tensor clamp_max(const Tensor& input, double maximum);
DL_API Tensor sqrt(const Tensor& input);
DL_API Tensor tanh(const Tensor& input);
DL_API Tensor sigmoid(const Tensor& input);
/// 负值映射为零，其余值保持原值；前向保留 NaN。
/// 反向在非正输出位置取零，其余位置传递上游梯度，零点的导数约定为零。
DL_API Tensor relu(const Tensor& input);
DL_API Tensor gelu(const Tensor& input);
/// Softplus 计算 log(1+exp(x))，通过分段平移保持有限输入的数值稳定。
DL_API Tensor softplus(const Tensor& input);
/// x * sigmoid(x) 的融合实现，反向保存输入并直接计算对应 VJP。
DL_API Tensor silu(const Tensor& input);
/// 非正输入乘以 negative_slope，零点的导数也取该斜率。
DL_API Tensor leaky_relu(const Tensor& input, double negative_slope = 0.01);

/// 同 dtype 输入按末尾维广播，比较结果为不参与求导的 bool 张量。
/// 比较保留输入类型的精度，支持 float32、float64、int64 和 bool。
DL_API Tensor eq(const Tensor& a, const Tensor& b);
DL_API Tensor ne(const Tensor& a, const Tensor& b);
DL_API Tensor lt(const Tensor& a, const Tensor& b);
DL_API Tensor le(const Tensor& a, const Tensor& b);
DL_API Tensor gt(const Tensor& a, const Tensor& b);
DL_API Tensor ge(const Tensor& a, const Tensor& b);

DL_API Tensor operator+(const Tensor& a, const Tensor& b);
DL_API Tensor operator-(const Tensor& a, const Tensor& b);
DL_API Tensor operator*(const Tensor& a, const Tensor& b);
DL_API Tensor operator/(const Tensor& a, const Tensor& b);
DL_API Tensor operator-(const Tensor& input);
DL_API Tensor operator+(const Tensor& input, double value);
DL_API Tensor operator-(const Tensor& input, double value);
DL_API Tensor operator*(const Tensor& input, double value);
DL_API Tensor operator/(const Tensor& input, double value);
DL_API Tensor operator+(double value, const Tensor& input);
DL_API Tensor operator-(double value, const Tensor& input);
DL_API Tensor operator*(double value, const Tensor& input);
DL_API Tensor operator/(double value, const Tensor& input);

/// axes 为互异的轴；空 axes 表示恒等归约，无 axes 的重载归约全部维度。
/// 恒等归约复制逻辑数据，保留符号零并直接传递梯度；keep 不改变这一分支的形状。
/// sum 的空归约结果为零，mean 的归约元素数为正。
DL_API Tensor sum(const Tensor& input);
DL_API Tensor sum(const Tensor& input, Shape axes, KeepDim keep = KeepDim::no);
DL_API Tensor mean(const Tensor& input);
DL_API Tensor mean(const Tensor& input, Shape axes, KeepDim keep = KeepDim::no);

/// 沿非空归约组取最大值；并列最大位置均分该组的上游梯度。
/// axes 的空列表表示恒等归约，无 axes 的重载归约全部维度；NaN 随最大值传播。
DL_API Tensor amax(const Tensor& input);
DL_API Tensor amax(const Tensor& input, Shape axes, KeepDim keep = KeepDim::no);
DL_API Tensor amin(const Tensor& input);
DL_API Tensor amin(const Tensor& input, Shape axes, KeepDim keep = KeepDim::no);
/// 非空归约组返回第一个极值位置，NaN 优先于普通值；结果为不求导的 int64。
/// 无 axis 时按逻辑行优先顺序返回全局位置；支持四种 dtype 和非连续布局。
DL_API Tensor argmax(const Tensor& input);
DL_API Tensor argmax(const Tensor& input, Index axis, KeepDim keep = KeepDim::no);
DL_API Tensor argmin(const Tensor& input);
DL_API Tensor argmin(const Tensor& input, Index axis, KeepDim keep = KeepDim::no);
/// 使用最大值平移计算 log(sum(exp(input)))，空归约组的结果为负无穷。
DL_API Tensor logsumexp(const Tensor& input);
DL_API Tensor logsumexp(const Tensor& input, Shape axes, KeepDim keep = KeepDim::no);
/// 方差分母为 N-correction，每组元素数 N 为正，0 <= correction < N。
/// 空 axes 将每个元素视为一组；标准差在常数组上的输入梯度取零。
DL_API Tensor var(const Tensor& input, double correction = 1.0);
DL_API Tensor var(const Tensor& input, Shape axes, double correction = 1.0, KeepDim keep = KeepDim::no);
DL_API Tensor stddev(const Tensor& input, double correction = 1.0);
DL_API Tensor stddev(const Tensor& input, Shape axes, double correction = 1.0, KeepDim keep = KeepDim::no);
/// 向量范数沿所选轴计算，其他轴形成批维；order 为正数、零或正负无穷。
/// 零阶统计非零元素数，正负无穷阶取绝对值极值；有限正阶使用缩放后的幂和。
DL_API Tensor vector_norm(const Tensor& input, double order = 2.0);
DL_API Tensor vector_norm(const Tensor& input, Shape axes, double order = 2.0, KeepDim keep = KeepDim::no);
/// 花括号列表始终表示轴列表，单个数值参数表示 correction 或 order。
inline Tensor var(const Tensor& input, std::initializer_list<Index> axes, double correction = 1.0, KeepDim keep = KeepDim::no) {
    return var(input, Shape(axes), correction, keep);
}
inline Tensor stddev(const Tensor& input, std::initializer_list<Index> axes, double correction = 1.0, KeepDim keep = KeepDim::no) {
    return stddev(input, Shape(axes), correction, keep);
}
inline Tensor vector_norm(const Tensor& input, std::initializer_list<Index> axes, double order = 2.0, KeepDim keep = KeepDim::no) {
    return vector_norm(input, Shape(axes), order, keep);
}

/// 最后两维相乘，前导批维按广播规则组合；输入的 rank 至少为一。
/// 左向量临时增加行维，右向量临时增加列维，输出移除对应维；两向量得到标量。
/// f16/bf16 输入使用 FP32 累积和输出，梯度同样使用 FP32；AutocastGuard 为 FP32 输入选择低精度 GEMM。
DL_API Tensor matmul(const Tensor& a, const Tensor& b);

/// 对末尾特征计算 input * weight^T + bias，保留所有前导维度。
/// input 为 [..., in]，weight 为 [out, in]，可选 bias 为 [out]；沿用 matmul 的精度策略。
/// 前导维合并为二维矩阵，偏置原地加入新结果，一个求导节点将三种梯度直接累计到目标。
DL_API Tensor linear(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias = std::nullopt);

/// 合并 first 到 last 的连续轴区间，负轴从末尾计数，归一化后 first <= last。
/// 单轴区间返回当前句柄；多轴区间按 reshape 规则共享或复制存储。
/// 标量的轴参数取 0 或 -1，结果为长度一的视图；默认合并全部维度。
DL_API Tensor flatten(const Tensor& input, Index first = 0, Index last = -1);
inline Tensor unsqueeze(const Tensor& input, Index axis) { return input.unsqueeze(axis); }
inline Tensor squeeze(const Tensor& input) { return input.squeeze(); }
inline Tensor squeeze(const Tensor& input, Index axis) { return input.squeeze(axis); }
inline Tensor select(const Tensor& input, Index axis, Index index) { return input.select(axis, index); }
inline Tensor narrow(const Tensor& input, Index axis, Index start, Index length) { return input.narrow(axis, start, length); }
inline std::vector<Tensor> split(const Tensor& input, Index size, Index axis = 0) { return input.split(size, axis); }
inline std::vector<Tensor> split(const Tensor& input, Shape sizes, Index axis = 0) { return input.split(std::move(sizes), axis); }
inline std::vector<Tensor> chunk(const Tensor& input, Index chunks, Index axis = 0) { return input.chunk(chunks, axis); }
inline std::vector<Tensor> unbind(const Tensor& input, Index axis = 0) { return input.unbind(axis); }
inline Tensor repeat(const Tensor& input, Shape repeats) { return input.repeat(std::move(repeats)); }

/// 沿指定非空轴计算稳定的归一化指数，前向与反向分别在一次行归约内完成。
/// 低精度输入直接读取，计算、输出与梯度保持 FP32；CPU f64 沿用双精度。
DL_API Tensor softmax(const Tensor& input, Index axis);
/// 直接从平移后的输入计算对数概率，反向使用 g-exp(output)*sum(g)，共用 softmax 的精度与布局规则。
DL_API Tensor log_softmax(const Tensor& input, Index axis);

/// 根据 bool mask 在两个同 dtype 输入间选择，三个输入共同广播；浮点输入参与求导。
DL_API Tensor where(const Tensor& mask, const Tensor& a, const Tensor& b);
DL_API Tensor where(const Tensor& mask, const Tensor& input, double other);

/// 用一维 int64 索引选择指定轴，重复索引在反向中累计到原位置。
DL_API Tensor index_select(const Tensor& input, Index axis, const Tensor& indices);

/// 从浮点 weight [vocabulary, dimensions] 中读取 int64 indices 指定的行。
/// indices 可为任意形状，值位于词表范围内；输出在索引形状末尾增加 dimensions。
/// 索引和权重同设备，重复索引的稠密梯度累计到相应权重行。
DL_API Tensor embedding(const Tensor& indices, const Tensor& weight);

/// input 与 int64 indices 具有相同的正 rank，结果形状等于 indices。
/// 非选择轴上 indices.size <= input.size，索引值位于 input 的选择轴范围内。
/// 数据读取支持全部 dtype，浮点输入按相应设备的求导范围记录 VJP。
DL_API Tensor gather(const Tensor& input, Index axis, const Tensor& indices);

/// 从 input 的独立副本开始，按 int64 indices 将 source 的元素相加到目标位置。
/// 三者 rank 相同且为正；indices 各维不超过 source，非选择轴不超过 input。
/// 索引值位于 input 的选择轴范围内，indices 覆盖范围之外的 source 元素不参与计算。
/// input 与 source 使用相同浮点 dtype，重复索引的贡献相加。
DL_API Tensor scatter_add(const Tensor& input, Index axis, const Tensor& indices, const Tensor& source);

/// 非空列表沿现有轴拼接，输入同 dtype、同设备，其他维度相同；结果拥有连续存储。
DL_API Tensor concat(std::span<const Tensor> inputs, Index axis = 0);
/// 同形状的非空列表沿新轴排列，标量也可以组成一维张量。
DL_API Tensor stack(std::span<const Tensor> inputs, Index axis = 0);
inline Tensor concat(std::initializer_list<Tensor> inputs, Index axis = 0) {
    return concat(std::span<const Tensor>(inputs.begin(), inputs.size()), axis);
}
inline Tensor stack(std::initializer_list<Tensor> inputs, Index axis = 0) {
    return stack(std::span<const Tensor>(inputs.begin(), inputs.size()), axis);
}

/// logits 为 [classes] 或 [batch, classes, ...]，int64 labels 对应移除类别轴后的形状。
/// 类别数为正，标签位于类别范围；默认取平均，空标签的 mean 为 NaN、sum 为零。
/// 各归约共用逐行内核，保存 logits 与紧凑统计；低精度输入使用 FP32 输出和梯度。
DL_API Tensor cross_entropy(const Tensor& logits, const Tensor& labels);
DL_API Tensor cross_entropy(const Tensor& logits, const Tensor& labels, LossReduction reduction);
/// 忽略指定标签；none 在该位置返回零，mean 按有效标签数归约。
/// 有限 logits 下，全忽略的 sum 为零、mean 为 NaN，两者的输入梯度均为零。
DL_API Tensor cross_entropy(const Tensor& logits, const Tensor& labels,
                            LossReduction reduction, Index ignore_index);
/// NLL 接收对数概率，类别轴与标签形状沿用 cross_entropy 的约定。
DL_API Tensor nll_loss(const Tensor& log_probabilities, const Tensor& labels,
                     LossReduction reduction = LossReduction::mean);
DL_API Tensor nll_loss(const Tensor& log_probabilities, const Tensor& labels,
                       LossReduction reduction, Index ignore_index);
/// 逐元素回归损失沿用浮点广播规则；none 保留广播形状，mean 的损失项数为正。
DL_API Tensor mse_loss(const Tensor& prediction, const Tensor& target);
DL_API Tensor mse_loss(const Tensor& prediction, const Tensor& target, LossReduction reduction);
DL_API Tensor l1_loss(const Tensor& prediction, const Tensor& target,
                    LossReduction reduction = LossReduction::mean);
/// SmoothL1 在 |error| < beta 时使用 error^2/(2*beta)，beta=0 时为 L1。
/// Huber 的 delta 为正，相当于 delta*SmoothL1(beta=delta)。
DL_API Tensor smooth_l1_loss(const Tensor& prediction, const Tensor& target, double beta = 1.0,
                           LossReduction reduction = LossReduction::mean);
DL_API Tensor huber_loss(const Tensor& prediction, const Tensor& target, double delta = 1.0,
                       LossReduction reduction = LossReduction::mean);
/// 概率与 target 同形且位于 [0,1]；两项 log 的下界为 -100。
/// 概率梯度为 (p-target)/max(p*(1-p),1e-12)，target 梯度使用两项截断 log 的差。
DL_API Tensor binary_cross_entropy(const Tensor& probability, const Tensor& target,
                                  LossReduction reduction = LossReduction::mean);
/// logits 与 target 同形，target 位于 [0,1]；以稳定 Softplus 计算正负类别损失。
DL_API Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target,
                                              LossReduction reduction = LossReduction::mean);

/// NCHW 互相关，weight 为 [out, in/groups, kh, kw]，bias 为可选的 [out]。
/// groups 整除输入与输出通道；stride、dilation 为正，padding 为非负对称填充。
/// 有效卷积核覆盖在填充后的空间范围内，输出高宽为正；batch 可以为零。
/// autocast 可选择 f16/bf16 输入；低精度卷积采用 FP32 累积、输出和叶梯度。
/// cuDNN 低精度结果可具有通道连续的物理布局，逻辑维度仍为 NCHW。
DL_API Tensor conv2d(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias = std::nullopt,
               Conv2dOptions options = {});
/// [N,C,L] 或 [C,L] 上的一维互相关，weight 为 [out,in/groups,k]；沿用二维卷积的参数约定。
DL_API Tensor conv1d(const Tensor& input, const Tensor& weight, std::optional<Tensor> bias = std::nullopt,
                    Conv1dOptions options = {});

/// NCHW 空间池化。最大值并列时按行优先选择第一个最大值位置。
/// 最大池化传播 NaN，含多个 NaN 的窗口选择行优先的最后一个 NaN 位置承接梯度。
/// 最大值索引按求导记录需求保存；窗口反向直接向输入梯度目标累计贡献。
DL_API Tensor max_pool2d(const Tensor& input, Pool2dOptions options = {});
DL_API Tensor avg_pool2d(const Tensor& input, Pool2dOptions options = {}, bool count_include_pad = true);
DL_API Tensor max_pool1d(const Tensor& input, Pool1dOptions options = {});
DL_API Tensor avg_pool1d(const Tensor& input, Pool1dOptions options = {}, bool count_include_pad = true);
/// 将正长度的空间维归约到指定正尺寸；分区边界按 floor/ceil 计算，重叠位置累加梯度。
/// 一维接受 [N,C,L]/[C,L]，二维接受 [N,C,H,W]/[C,H,W]；batch 可以为零。
/// 二维计算直接读取最后两个空间轴，反向把区域贡献追加到输入梯度目标。
DL_API Tensor adaptive_avg_pool1d(const Tensor& input, Index output_size);
DL_API Tensor adaptive_avg_pool2d(const Tensor& input, Size2 output_size);
/// 对 [N,C,H,W] 或 [C,H,W] 浮点图像指定正输出尺寸，输入空间维为正。
/// bilinear 的 align_corners 选择角像素中心对齐，否则采用半像素坐标和边缘延拓。
/// 反向使用相同源坐标与权重，直接累计到输入梯度目标。
DL_API Tensor interpolate(const Tensor& input, Size2 output_size,
                          InterpolationMode mode = InterpolationMode::nearest, bool align_corners = false);

/// normalized_shape 非空、各维为正，并与输入末尾形状一致；前导维可含零长度维。
/// 使用总体方差，epsilon 加到方差后开方；可选 weight、bias 与 normalized_shape 同形。
/// 输入与仿射张量共用设备；仿射参数使用输入或累积 dtype，低精度输入的计算与输出为 FP32。
/// 统计和仿射共用融合内核，非连续布局与共享参数沿用普通算子的梯度规则。
DL_API Tensor layer_norm(const Tensor& input, const Shape& normalized_shape,
                     std::optional<Tensor> weight = std::nullopt, std::optional<Tensor> bias = std::nullopt,
                     double epsilon = 1e-5);

/// [C] 的运行统计，dtype 为输入的累积类型；均值和方差作为普通张量显式传递。
struct BatchNormState { Tensor mean; Tensor variance; };
struct BatchNormResult { Tensor output; std::optional<BatchNormState> state; };

/// 对 [N,C,...] 按 batch 和空间轴归一化；使用批统计时每通道元素数大于一。
/// 可选 weight/bias 为 [C]，与输入同设备，使用输入或累积 dtype；输出采用累积 dtype。
/// 无 state 时 train/eval 都使用当前批统计；提供 state 时，训练返回新的运行统计。
/// momentum 是当前批权重，运行方差使用无偏估计；状态不参与求导，原状态保持原值。
/// eval 返回脱离求导历史的原状态别名，训练返回独立、连续、可写的新状态。
/// 一个求导节点共用通道归约，按需求保存原输入和统计量，反向重算标准化值。
DL_API BatchNormResult batch_norm(const Tensor& input, std::optional<BatchNormState> state = std::nullopt,
    std::optional<Tensor> weight = std::nullopt, std::optional<Tensor> bias = std::nullopt,
    bool training = true, double momentum = 0.1, double epsilon = 1e-5);

/// 使用同一归一化计算并直接更新提供的运行统计，保留状态张量身份和布局。
/// state 的两份存储可写且互不重叠，与输入及仿射参数分开；eval 只读取状态。
/// NoGrad 只控制图记录，训练中的统计更新仍然执行。
DL_API Tensor batch_norm_(const Tensor& input, BatchNormState state,
    std::optional<Tensor> weight = std::nullopt, std::optional<Tensor> bias = std::nullopt,
    bool training = true, double momentum = 0.1, double epsilon = 1e-5);

/// 在 [N,C,...] 的每个样本内按通道分组，groups 整除 C；每组元素数为正。
/// 使用总体方差，可选 weight/bias 为 [C]；train/eval 均由当前输入计算统计量。
DL_API Tensor group_norm(const Tensor& input, Index groups, std::optional<Tensor> weight = std::nullopt,
                         std::optional<Tensor> bias = std::nullopt, double epsilon = 1e-5);
/// 对末尾 normalized_shape 计算 x/sqrt(mean(x*x)+epsilon)，可选 weight 逐元素缩放。
/// epsilon 默认取累积 dtype 的机器精度，f16/bf16/f32 为 FP32，CPU f64 为 FP64。
/// 归一化形状、融合计算与布局沿用 layer_norm。
DL_API Tensor rms_norm(const Tensor& input, const Shape& normalized_shape,
                       std::optional<Tensor> weight = std::nullopt, std::optional<double> epsilon = std::nullopt);

/// 倒置 Dropout 的 probability 位于 [0,1]；eval 或零概率时返回输入句柄。
DL_API Tensor dropout(const Tensor& input, double probability = 0.5, bool training = true);
/// 显式随机流只在本次前向采样时使用；反向保存掩码，不再访问生成器。
DL_API Tensor dropout(const Tensor& input, double probability, Generator& generator, bool training = true);
/// 按显式 key 生成掩码，不推进随机状态；反向继续读取本次保存的掩码。
DL_API Tensor dropout(const Tensor& input, double probability, RandomKey key, bool training = true);

/// Q/K/V 的最后两维分别为 [L,D]、[S,D]、[S,Dv]，前导维按广播规则组合。
/// bool keep_mask 可广播到分数矩阵，true 保留位置，每行至少保留一个位置。
DL_API Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::optional<Tensor> keep_mask = std::nullopt,
                                    double dropout_probability = 0.0);
/// 显式 key 只用于注意力概率的 Dropout，矩阵乘与反向沿用相同实现。
DL_API Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::optional<Tensor> keep_mask, double dropout_probability,
                                    RandomKey random_key);

/// 选项入口显式选择后端，causal 与 keep_mask 共同决定保留位置。
/// cudnn 接收四维 [B,H,L,D]，FP16/BF16 输入或 CUDA autocast 下的 FP32 输入，返回 FP32。
/// 融合路径保存 Q/K/V、低精度输出及逐行统计，并在反向重建概率；dropout 位于 [0,1)。
/// CUDA/cuDNN 的设备、形状和掩码范围见注意力说明，未启用的后端直接报告不可用。
DL_API Tensor scaled_dot_product_attention(AttentionComputeOptions options,
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::optional<Tensor> keep_mask = std::nullopt);
DL_API Tensor scaled_dot_product_attention(AttentionComputeOptions options,
    const Tensor& query, const Tensor& key, const Tensor& value, RandomKey random_key,
    std::optional<Tensor> keep_mask = std::nullopt);

/// 上左对齐的因果掩码，key_position <= query_position 时为 true；可在多个前向间复用。
DL_API Tensor causal_mask(Index query_length, Index key_length, Device device = Device::cpu());

/// LSTM 的隐藏状态与记忆状态具有相同形状，均参与普通的一阶求导。
struct LSTMState { Tensor hidden; Tensor cell; };

/// 单步循环接受 [batch,input_size] 或 [input_size]，hidden 对应 [...,hidden_size]。
/// 输入权重为 [gates*hidden_size,input_size]，循环权重为 [gates*hidden_size,hidden_size]。
/// RNN/GRU/LSTM 分别使用 1/3/4 组门；偏置为可选的 [gates*hidden_size]。
DL_API Tensor rnn_cell(const Tensor& input, const Tensor& hidden, const Tensor& weight_ih, const Tensor& weight_hh,
                       std::optional<Tensor> bias_ih = std::nullopt, std::optional<Tensor> bias_hh = std::nullopt,
                       RNNActivation activation = RNNActivation::tanh);
/// GRU 门顺序为 reset/update/new，reset 乘在隐藏投影及其偏置之后。
DL_API Tensor gru_cell(const Tensor& input, const Tensor& hidden, const Tensor& weight_ih, const Tensor& weight_hh,
                       std::optional<Tensor> bias_ih = std::nullopt, std::optional<Tensor> bias_hh = std::nullopt);
/// LSTM 门顺序为 input/forget/candidate/output。
DL_API LSTMState lstm_cell(const Tensor& input, const LSTMState& state, const Tensor& weight_ih, const Tensor& weight_hh,
                           std::optional<Tensor> bias_ih = std::nullopt, std::optional<Tensor> bias_hh = std::nullopt);

} // namespace dl
