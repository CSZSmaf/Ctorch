#pragma once

#include "dl/autograd.h"

#include <variant>

namespace dl::optim {

struct SGDOptions {
    double learning_rate = 0.01;
    double momentum = 0.0;
    double dampening = 0.0;
    double weight_decay = 0.0;
    bool nesterov = false;
};

struct AdamWOptions {
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.01;
};

struct AdamOptions {
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
};

struct RMSpropOptions {
    double learning_rate = 0.01;
    double alpha = 0.99;
    double epsilon = 1e-8;
    double weight_decay = 0.0;
    double momentum = 0.0;
    bool centered = false;
};

struct AdagradOptions {
    double learning_rate = 0.01;
    double learning_rate_decay = 0.0;
    double weight_decay = 0.0;
    double initial_accumulator = 0.0;
    double epsilon = 1e-10;
};

/// 每组可覆盖学习率和权重衰减，其余配置由优化器统一管理。
/// 不同组使用不同参数身份；组内的重复句柄只登记一次。
/// 更新目标使用可写存储，外部只读值通过 clone 建立独立副本。
struct ParameterGroup {
    std::vector<Tensor> parameters;
    std::optional<double> learning_rate;
    std::optional<double> weight_decay;
};

struct ParameterGroupState {
    std::vector<std::size_t> parameters;
    double learning_rate = 0.0;
    double weight_decay = 0.0;
};

struct SGDState {
    SGDOptions options;
    std::vector<std::optional<Tensor>> velocity;
    std::vector<ParameterGroupState> groups;
};

struct AdamMoments {
    Tensor first;
    Tensor second;
    Index step = 0;
};
using AdamWMoments = AdamMoments;

struct RMSpropBuffers {
    Tensor square_average;
    std::optional<Tensor> momentum;
    std::optional<Tensor> gradient_average;
    Index step = 0;
};

struct AdagradAccumulator {
    Tensor sum;
    Index step = 0;
};

struct SGDResult {
    Tensor parameter;
    std::optional<Tensor> velocity;
};

struct AdamWResult {
    Tensor parameter;
    AdamMoments moments;
};
using AdamResult = AdamWResult;

struct RMSpropResult {
    Tensor parameter;
    RMSpropBuffers buffers;
};

struct AdagradResult {
    Tensor parameter;
    AdagradAccumulator accumulator;
};

/// 显式梯度驱动一次更新，参数、梯度和状态具有相同 shape、dtype 和设备。
/// CPU 使用 f32/f64，CUDA 使用 f32；只读输入和非连续输入按逻辑元素读取。
/// 新参数与状态拥有独立连续存储，不记录更新过程的求导历史；新参数成为空梯度叶，
/// 保留原参数的 requires_grad。缺失梯度由调用方跳过，零梯度仍执行更新。
/// SGD 的空 velocity 从当前方向建立，已有 velocity 在 momentum=0 时复制保留。
DL_API SGDResult sgd(const Tensor& parameter, const Tensor& gradient,
    const std::optional<Tensor>& velocity = std::nullopt, SGDOptions options = {});
DL_API AdamWResult adamw(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdamMoments>& moments = std::nullopt, AdamWOptions options = {});
DL_API AdamResult adam(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdamMoments>& moments = std::nullopt, AdamOptions options = {});
/// RMSprop 的可选动量和梯度均值在启用时从零建立，停用时保留旧值以供后续继续。
DL_API RMSpropResult rmsprop(const Tensor& parameter, const Tensor& gradient,
    const std::optional<RMSpropBuffers>& buffers = std::nullopt, RMSpropOptions options = {});
DL_API AdagradResult adagrad(const Tensor& parameter, const Tensor& gradient,
    const std::optional<AdagradAccumulator>& accumulator = std::nullopt, AdagradOptions options = {});

/// 原地入口保留参数及已有缓冲的身份，首次使用时填充空状态槽。
/// 参数是已完成相关反向图的可写叶；状态为无求导历史的可写张量。
/// 写入目标各自不重叠，并与只读梯度分开；本次更新保留参数的现有梯度槽。
DL_API void sgd_(Tensor parameter, const Tensor& gradient,
    std::optional<Tensor>& velocity, SGDOptions options = {});
DL_API void adamw_(Tensor parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, AdamWOptions options = {});
DL_API void adam_(Tensor parameter, const Tensor& gradient,
    std::optional<AdamMoments>& moments, AdamOptions options = {});
DL_API void rmsprop_(Tensor parameter, const Tensor& gradient,
    std::optional<RMSpropBuffers>& buffers, RMSpropOptions options = {});
DL_API void adagrad_(Tensor parameter, const Tensor& gradient,
    std::optional<AdagradAccumulator>& accumulator, AdagradOptions options = {});

struct AdamWState {
    AdamWOptions options;
    std::vector<std::optional<AdamMoments>> moments;
    std::vector<ParameterGroupState> groups;
};

struct AdamState {
    AdamOptions options;
    std::vector<std::optional<AdamMoments>> moments;
    std::vector<ParameterGroupState> groups;
};

struct RMSpropState {
    RMSpropOptions options;
    std::vector<std::optional<RMSpropBuffers>> buffers;
    std::vector<ParameterGroupState> groups;
};

struct AdagradState {
    AdagradOptions options;
    std::vector<std::optional<AdagradAccumulator>> accumulators;
    std::vector<ParameterGroupState> groups;
};

using OptimizerState = std::variant<SGDState, AdamWState, AdamState, RMSpropState, AdagradState>;

/// 优化器持有连续浮点叶参数，按身份去重，并在相关前向图反向结束后更新。
/// 各参数及状态的写入区间互不重叠；共享权重通过同一 Tensor 身份登记。
/// 五种算法共用批量调度，CUDA 合并连续参数的提交，每个参数保留自己的状态和组配置。
class DL_API Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> parameters, double learning_rate = 0.0, double weight_decay = 0.0);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zero_grad();
    const std::vector<Tensor>& parameters() const { return parameters_; }
    std::span<const ParameterGroupState> groups() const { return groups_; }
    double learning_rate(std::size_t group = 0) const { return groups_[group].learning_rate; }
    double weight_decay(std::size_t group = 0) const { return groups_[group].weight_decay; }
    void set_learning_rate(double value, std::size_t group = 0) { groups_[group].learning_rate = value; }
    void set_weight_decay(double value, std::size_t group = 0) { groups_[group].weight_decay = value; }

    /// 状态槽与去重后的参数一一对应，空槽表示尚未建立对应动量。
    /// 返回值共享当前缓冲存储；加载时按逻辑元素复制每个非空槽，并恢复算法配置。
    /// 输入缓冲可为视图，shape 与 dtype 对应相应参数；算法与槽位数对应当前优化器。
    /// 加载后的各缓冲独立、连续、无求导历史，位于相应参数的设备上。
    /// options 保存算法默认配置，groups 保存各组当前配置；空 groups 表示默认配置的一组参数。
    virtual OptimizerState state_dict() const = 0;
    virtual void load_state_dict(const OptimizerState& state) = 0;

protected:
    Optimizer(std::vector<ParameterGroup> groups, double learning_rate, double weight_decay);
    void load_groups(const std::vector<ParameterGroupState>& groups, double learning_rate, double weight_decay);
    std::vector<Tensor> parameters_;
    std::vector<ParameterGroupState> groups_;
};

class DL_API SGD : public Optimizer {
public:
    explicit SGD(std::vector<Tensor> parameters, SGDOptions options = {});
    explicit SGD(std::vector<ParameterGroup> groups, SGDOptions options = {});
    void step() override;
    OptimizerState state_dict() const override;
    void load_state_dict(const OptimizerState& state) override;

private:
    SGDOptions options_;
    std::vector<std::optional<Tensor>> velocity_;
};

class DL_API AdamW : public Optimizer {
public:
    explicit AdamW(std::vector<Tensor> parameters, AdamWOptions options = {});
    explicit AdamW(std::vector<ParameterGroup> groups, AdamWOptions options = {});
    void step() override;
    OptimizerState state_dict() const override;
    void load_state_dict(const OptimizerState& state) override;

private:
    AdamWOptions options_;
    std::vector<std::optional<AdamMoments>> states_;
};

/// Adam 的权重衰减进入梯度和矩估计；AdamW 在参数更新时独立施加衰减。
class DL_API Adam : public Optimizer {
public:
    explicit Adam(std::vector<Tensor> parameters, AdamOptions options = {});
    explicit Adam(std::vector<ParameterGroup> groups, AdamOptions options = {});
    void step() override;
    OptimizerState state_dict() const override;
    void load_state_dict(const OptimizerState& state) override;

private:
    AdamOptions options_;
    std::vector<std::optional<AdamMoments>> states_;
};

class DL_API RMSprop : public Optimizer {
public:
    explicit RMSprop(std::vector<Tensor> parameters, RMSpropOptions options = {});
    explicit RMSprop(std::vector<ParameterGroup> groups, RMSpropOptions options = {});
    void step() override;
    OptimizerState state_dict() const override;
    void load_state_dict(const OptimizerState& state) override;

private:
    RMSpropOptions options_;
    std::vector<std::optional<RMSpropBuffers>> states_;
};

class DL_API Adagrad : public Optimizer {
public:
    explicit Adagrad(std::vector<Tensor> parameters, AdagradOptions options = {});
    explicit Adagrad(std::vector<ParameterGroup> groups, AdagradOptions options = {});
    void step() override;
    OptimizerState state_dict() const override;
    void load_state_dict(const OptimizerState& state) override;

private:
    AdagradOptions options_;
    std::vector<std::optional<AdagradAccumulator>> states_;
};

/// 按保存的算法创建优化器，参数顺序对应状态槽，并将缓冲复制到各参数设备。
/// 返回的优化器保留传入参数身份，适合绑定恢复得到的新叶集合。
DL_API std::unique_ptr<Optimizer> from_state(std::vector<Tensor> parameters, const OptimizerState& state);

/// 在 backward 与 step 之间裁剪叶梯度，重复参数只处理一次，无梯度参数保持原状。
/// max_norm 和 value 为非负值。范数接口接受同 dtype、同设备的浮点参数，order 为正数或正无穷。
/// 范数接口返回裁剪前的总范数。
/// 总范数留在原设备，max_norm/(norm+1e-6) 的比例最多为一。
/// 数值层直接归约梯度并原地缩放，CUDA 合并连续梯度的遍历。
DL_API Tensor clip_grad_norm(std::span<const Tensor> parameters, double max_norm, double order = 2.0);
DL_API void clip_grad_value(std::span<const Tensor> parameters, double value);

struct ClipGradNormResult {
    Gradients gradients;
    Tensor total_norm;
};

/// 直接处理梯度值，保留空项与重复身份，计算过程不记录求导图。
/// 纯入口按身份复制独立连续结果；原地入口处理可写值，其他身份的逻辑元素彼此不重叠。
/// 同身份只参与一次范数和写入。CPU 使用 f32/f64，CUDA 使用 f32；裁剪界限非负。
/// 范数入口的非空项同 dtype、设备，order 为正数或正无穷，总范数留在该设备。
/// 全空列表返回 CPU f32 标量零；空张量对整体范数的贡献为零。
DL_API ClipGradNormResult clip_gradients_norm(std::span<const std::optional<Tensor>> gradients,
    double max_norm, double order = 2.0);
DL_API Tensor clip_gradients_norm_(std::span<const std::optional<Tensor>> gradients,
    double max_norm, double order = 2.0);
DL_API Gradients clip_gradients_value(std::span<const std::optional<Tensor>> gradients, double value);
DL_API void clip_gradients_value_(std::span<const std::optional<Tensor>> gradients, double value);

} // namespace dl::optim
