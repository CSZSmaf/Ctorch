#pragma once

#include "dl/ops.h"
#include "dl/random.h"

#include <string>
#include <utility>

namespace dl {

using NamedTensor = std::pair<std::string, Tensor>;
using StateDict = std::vector<NamedTensor>;

namespace nn {

class FunctionalModule;

using TrainingModes = std::vector<std::pair<std::string, bool>>;
using GradientModes = std::vector<std::pair<std::string, bool>>;

/// 模块登记参数、buffer 与子模块，遍历结果保持名称顺序并按张量身份去重。
class DL_API Module {
public:
    virtual ~Module() = default;
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    std::vector<Tensor> parameters() const;
    StateDict named_parameters() const;
    /// 只选择当前需要梯度的参数；全部参数入口仍保留冻结参数及原有遍历顺序。
    std::vector<Tensor> trainable_parameters() const;
    StateDict named_trainable_parameters() const;
    StateDict named_buffers() const;

    /// 在完整前向图之间递归冻结或解冻参数，按身份去重；train/eval 和 buffer 保持原样。
    void requires_grad_(bool enabled = true);
    /// 使用 named_parameters 的规范名称保存逐参数状态，共享权重只出现一次。
    GradientModes gradient_modes() const;
    void load_gradient_modes(std::span<const std::pair<std::string, bool>> modes);

    /// 状态表中的张量共享当前存储。
    StateDict state_dict() const;
    /// 加载将同名同形状同 dtype 的数值写入既有可写对象。
    /// 与目标存储共享的输入先保存调用时的值，参数互换与视图写回保持同一数值语义。
    void load_state_dict(std::span<const NamedTensor> state);

    /// 在前向图之间迁移参数、梯度与 buffer，保留参数身份和权重绑定。
    /// 模型的设备在创建优化器前确定。
    void to(Device device);

    void train(bool enabled = true);
    void eval() { train(false); }
    bool is_training() const;
    /// 根模块名称为空，各子模块使用点分路径，分别保留 train/eval 选择。
    TrainingModes training_modes() const;
    void load_training_modes(std::span<const std::pair<std::string, bool>> modes);

protected:
    /// 前向读取登记参数的入口；普通调用返回自身句柄，函数式调用返回相应的外部绑定。
    /// 嵌套作用域按模块归属选择最近绑定，参数、buffer、模式与随机 key 使用同一归属。
    /// 返回引用在当前调用内使用，复制 Tensor 句柄可独立保持值的生命周期。
    const Tensor& bound(const Tensor& value) const;
    std::optional<Tensor> bound(const std::optional<Tensor>& value) const;
    /// buffer 使用独立状态槽，函数式调用中的写入落在本次复制得到的存储上。
    const Tensor& bound_buffer(const Tensor& value) const;
    /// 函数式调用按模块槽和调用内序号派生 key；普通调用为空并沿用 Generator。
    std::optional<RandomKey> next_random_key() const;
    Tensor apply_dropout(const Tensor& input, double probability) const;

    /// 参数为存储可写的浮点叶张量，登记后保留传入身份及当前求导状态。
    Tensor register_parameter(std::string name, Tensor value);
    Tensor register_buffer(std::string name, Tensor value);
    /// 子模块可以在外部构造后登记；父模块持有它的共享所有权。
    void register_module(std::string name, std::shared_ptr<Module> child);

    template <class Child, class... Args>
    std::shared_ptr<Child> add_module(std::string name, Args&&... args) {
        auto child = std::make_shared<Child>(std::forward<Args>(args)...);
        register_module(std::move(name), child);
        return child;
    }

private:
    StateDict collect(bool buffers) const;

    StateDict parameters_;
    StateDict buffers_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
    bool training_ = true;
    friend class FunctionalModule;
};

/// 单输入单输出模块的统一调用接口，Sequential 通过它组合不同的层。
class DL_API UnaryModule : public Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input) { return forward(input); }
};

/// 登记权重与可选偏置，前向共用 linear 的末尾特征投影。
class DL_API Linear : public UnaryModule {
public:
    Linear(Index input_features, Index output_features, bool bias = true, TensorOptions options = {});
    Linear(Tensor weight, std::optional<Tensor> bias = std::nullopt);
    Tensor forward(const Tensor& input) override;
    const Tensor& weight() const { return weight_; }
    const std::optional<Tensor>& bias() const { return bias_; }

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
};

class DL_API ReLU : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return relu(input); }
};

class DL_API GELU : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return gelu(input); }
};

class DL_API Sigmoid : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return sigmoid(input); }
};

class DL_API Tanh : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return tanh(input); }
};

class DL_API SiLU : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return silu(input); }
};

class DL_API Softplus : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override { return softplus(input); }
};

class DL_API LeakyReLU : public UnaryModule {
public:
    explicit LeakyReLU(double negative_slope = 0.01) : negative_slope_(negative_slope) {}
    Tensor forward(const Tensor& input) override { return leaky_relu(input, negative_slope_); }

private:
    double negative_slope_;
};

class DL_API Softmax : public UnaryModule {
public:
    explicit Softmax(Index axis) : axis_(axis) {}
    Tensor forward(const Tensor& input) override { return softmax(input, axis_); }

private:
    Index axis_;
};

class DL_API LogSoftmax : public UnaryModule {
public:
    explicit LogSoftmax(Index axis) : axis_(axis) {}
    Tensor forward(const Tensor& input) override { return log_softmax(input, axis_); }

private:
    Index axis_;
};

/// 保存待合并的轴区间，默认保留首个批维，前向调用 flatten。
class DL_API Flatten : public UnaryModule {
public:
    explicit Flatten(Index first = 1, Index last = -1) : first_(first), last_(last) {}
    Tensor forward(const Tensor& input) override;

private:
    Index first_;
    Index last_;
};

class DL_API Sequential : public UnaryModule {
public:
    void append(std::string name, std::shared_ptr<UnaryModule> layer);
    template <class Layer, class... Args>
    std::shared_ptr<Layer> append(std::string name, Args&&... args) {
        auto layer = std::make_shared<Layer>(std::forward<Args>(args)...);
        append(std::move(name), layer);
        return layer;
    }
    Tensor forward(const Tensor& input) override;

private:
    std::vector<std::shared_ptr<UnaryModule>> layers_;
};

/// 登记词表权重，前向共用 embedding 的行读取和重复索引梯度累计。
class DL_API Embedding : public UnaryModule {
public:
    Embedding(Index vocabulary, Index dimensions, TensorOptions options = {});
    explicit Embedding(Tensor weight);
    Tensor forward(const Tensor& indices) override;
    const Tensor& weight() const { return weight_; }

private:
    Tensor weight_;
};

/// 登记逐元素仿射参数，前向共用 layer_norm 对末尾 normalized_shape 维归一化。
class DL_API LayerNorm : public UnaryModule {
public:
    explicit LayerNorm(Shape normalized_shape, double epsilon = 1e-5, bool affine = true,
                       TensorOptions options = {});
    Tensor forward(const Tensor& input) override;

private:
    Shape normalized_shape_;
    double epsilon_;
    std::optional<Tensor> weight_;
    std::optional<Tensor> bias_;
};

/// 训练时按保留概率缩放采样结果，eval 模式下保持输入不变。
class DL_API Dropout : public UnaryModule {
public:
    explicit Dropout(double probability = 0.5) : probability_(probability) {}
    Tensor forward(const Tensor& input) override;

private:
    double probability_;
};

class DL_API Conv1d : public UnaryModule {
public:
    Conv1d(Index input_channels, Index output_channels, Index kernel,
           Conv1dOptions convolution = {}, bool bias = true, TensorOptions options = {});
    Conv1d(Tensor weight, std::optional<Tensor> bias = std::nullopt, Conv1dOptions convolution = {});
    Tensor forward(const Tensor& input) override;
    const Tensor& weight() const { return weight_; }
    const std::optional<Tensor>& bias() const { return bias_; }

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
    Conv1dOptions options_;
};

class DL_API Conv2d : public UnaryModule {
public:
    Conv2d(Index input_channels, Index output_channels, Size2 kernel,
            Conv2dOptions convolution = {}, bool bias = true, TensorOptions options = {});
    Conv2d(Tensor weight, std::optional<Tensor> bias = std::nullopt, Conv2dOptions convolution = {});
    Tensor forward(const Tensor& input) override;
    const Tensor& weight() const { return weight_; }
    const std::optional<Tensor>& bias() const { return bias_; }

private:
    Tensor weight_;
    std::optional<Tensor> bias_;
    Conv2dOptions options_;
};

class DL_API MaxPool2d : public UnaryModule {
public:
    explicit MaxPool2d(Pool2dOptions options = {}) : options_(options) {}
    Tensor forward(const Tensor& input) override { return max_pool2d(input, options_); }

private:
    Pool2dOptions options_;
};

class DL_API AvgPool2d : public UnaryModule {
public:
    explicit AvgPool2d(Pool2dOptions options = {}, bool count_include_pad = true)
        : options_(options), count_include_pad_(count_include_pad) {}
    Tensor forward(const Tensor& input) override { return avg_pool2d(input, options_, count_include_pad_); }

private:
    Pool2dOptions options_;
    bool count_include_pad_;
};

class DL_API MaxPool1d : public UnaryModule {
public:
    explicit MaxPool1d(Pool1dOptions options = {}) : options_(options) {}
    Tensor forward(const Tensor& input) override { return max_pool1d(input, options_); }
private:
    Pool1dOptions options_;
};

class DL_API AvgPool1d : public UnaryModule {
public:
    explicit AvgPool1d(Pool1dOptions options = {}, bool count_include_pad = true)
        : options_(options), count_include_pad_(count_include_pad) {}
    Tensor forward(const Tensor& input) override { return avg_pool1d(input, options_, count_include_pad_); }
private:
    Pool1dOptions options_;
    bool count_include_pad_;
};

class DL_API AdaptiveAvgPool1d : public UnaryModule {
public:
    explicit AdaptiveAvgPool1d(Index output_size) : output_size_(output_size) {}
    Tensor forward(const Tensor& input) override { return adaptive_avg_pool1d(input, output_size_); }
private:
    Index output_size_;
};

class DL_API AdaptiveAvgPool2d : public UnaryModule {
public:
    explicit AdaptiveAvgPool2d(Size2 output_size) : output_size_(output_size) {}
    Tensor forward(const Tensor& input) override { return adaptive_avg_pool2d(input, output_size_); }
private:
    Size2 output_size_;
};

class DL_API Upsample : public UnaryModule {
public:
    explicit Upsample(Size2 output_size, InterpolationMode mode = InterpolationMode::nearest, bool align_corners = false)
        : output_size_(output_size), mode_(mode), align_corners_(align_corners) {}
    Tensor forward(const Tensor& input) override { return interpolate(input, output_size_, mode_, align_corners_); }
private:
    Size2 output_size_;
    InterpolationMode mode_;
    bool align_corners_;
};

/// 按 batch 和全部空间轴统计每个通道，训练归一化使用总体方差，运行方差使用无偏估计。
/// 输入为 [N,C,...]，使用批统计时每通道元素数大于一；固定 momentum 为当前批的统计权重。
/// train/eval 选择统计来源，NoGrad 选择是否记录求导；训练中的 NoGrad 前向仍更新运行统计。
/// 运行 buffer 使用累积 dtype，前向复用 batch_norm_ 的原地状态入口。
class DL_API BatchNorm : public UnaryModule {
public:
    explicit BatchNorm(Index channels, BatchNormOptions normalization = {}, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
    void reset_running_stats();
    const std::optional<Tensor>& running_mean() const { return running_mean_; }
    const std::optional<Tensor>& running_variance() const { return running_variance_; }

private:
    Index channels_;
    BatchNormOptions options_;
    std::optional<Tensor> weight_, bias_;
    std::optional<Tensor> running_mean_, running_variance_;
};

/// [N,C] 或 [N,C,L] 的批归一化，共用 BatchNorm 的参数和运行统计。
class DL_API BatchNorm1d : public BatchNorm {
public:
    using BatchNorm::BatchNorm;
};

/// [N,C,H,W] 的批归一化。
class DL_API BatchNorm2d : public BatchNorm {
public:
    using BatchNorm::BatchNorm;
};

class DL_API GroupNorm : public UnaryModule {
public:
    GroupNorm(Index groups, Index channels, double epsilon = 1e-5, bool affine = true, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
private:
    Index groups_;
    double epsilon_;
    std::optional<Tensor> weight_, bias_;
};

class DL_API RMSNorm : public UnaryModule {
public:
    explicit RMSNorm(Shape normalized_shape, std::optional<double> epsilon = std::nullopt,
                     bool affine = true, TensorOptions options = {});
    Tensor forward(const Tensor& input) override;
private:
    Shape normalized_shape_;
    std::optional<double> epsilon_;
    std::optional<Tensor> weight_;
};

struct RecurrentResult {
    Tensor output;
    Tensor hidden;
    std::optional<Tensor> cell;
};

/// 输入为 [time,batch,features]，batch_first 选择 [batch,time,features]；[time,features] 表示单序列。
/// layers、hidden_size 和 time 为正。隐藏状态按 [layers*directions,batch,hidden_size] 排列，
/// 单序列省略 batch 轴；每层先正向后反向，双向输出在末轴拼接。
class DL_API Recurrent : public UnaryModule {
public:
    Tensor forward(const Tensor& input) override;
    /// 未提供的初始状态为零；RNN/GRU 的 cell 为空，LSTM 可分别传入 hidden 与 cell。
    /// 返回状态保留求导历史，跨训练段截断历史时由调用者显式 detach。
    RecurrentResult run(const Tensor& input, std::optional<Tensor> hidden = std::nullopt,
                         std::optional<Tensor> cell = std::nullopt);

protected:
    Recurrent(Index input_size, Index hidden_size, RecurrentKind kind, RecurrentOptions recurrent, TensorOptions options);

private:
    struct Weights {
        Tensor input, hidden;
        std::optional<Tensor> input_bias, hidden_bias;
    };
    Index hidden_size_;
    RecurrentKind kind_;
    RecurrentOptions options_;
    std::vector<Weights> weights_;
};

class DL_API RNN : public Recurrent {
public:
    RNN(Index input_size, Index hidden_size, RecurrentOptions recurrent = {},
        RNNActivation activation = RNNActivation::tanh, TensorOptions options = {});
};

class DL_API GRU : public Recurrent {
public:
    GRU(Index input_size, Index hidden_size, RecurrentOptions recurrent = {}, TensorOptions options = {});
};

class DL_API LSTM : public Recurrent {
public:
    LSTM(Index input_size, Index hidden_size, RecurrentOptions recurrent = {}, TensorOptions options = {});
};

/// 输入为 [batch, length, dimensions]，heads 整除 dimensions。
/// Q/K/V 的 batch 相同，K/V 的序列长度相同，Q 可以使用不同的序列长度。
/// 自注意力与交叉注意力共享投影层，所有 mask 均采用 true 表示保留的语义。
class DL_API MultiHeadAttention : public UnaryModule {
public:
    MultiHeadAttention(Index dimensions, Index heads, AttentionOptions attention = {}, TensorOptions options = {});
    /// 在完整前向图之间选择计算后端和上左对齐的因果关系，投影参数与 Dropout 概率保持原配置。
    void set_attention_backend(AttentionBackend backend, bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, std::optional<Tensor> keep_mask);
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value,
                    std::optional<Tensor> keep_mask = std::nullopt);

private:
    Index dimensions_, heads_;
    AttentionComputeOptions compute_;
    std::shared_ptr<Linear> query_, key_, value_, output_;
};

/// Pre-Norm 编码层依次组合注意力残差与 GELU 前馈残差。
class DL_API TransformerEncoderLayer : public UnaryModule {
public:
    TransformerEncoderLayer(Index dimensions, Index heads, Index feedforward,
                             double dropout_probability = 0.1, TensorOptions options = {});
    void set_attention_backend(AttentionBackend backend, bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, std::optional<Tensor> keep_mask);

private:
    std::shared_ptr<MultiHeadAttention> attention_;
    std::shared_ptr<LayerNorm> attention_norm_, feedforward_norm_;
    std::shared_ptr<Linear> expand_, contract_;
    std::shared_ptr<Dropout> attention_dropout_, hidden_dropout_, output_dropout_;
};

/// 每层拥有独立参数，所有层使用同一 mask，最后一次 LayerNorm 归一化堆叠结果。
class DL_API TransformerEncoder : public UnaryModule {
public:
    TransformerEncoder(Index layers, Index dimensions, Index heads, Index feedforward,
                        double dropout_probability = 0.1, TensorOptions options = {});
    /// 各层共用相同后端选择，融合因果计算无需单独创建方形 mask。
    void set_attention_backend(AttentionBackend backend, bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, std::optional<Tensor> keep_mask);

private:
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
    std::shared_ptr<LayerNorm> normalization_;
};

} // namespace nn
} // namespace dl
