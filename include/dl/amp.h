#pragma once

#include "dl/autograd.h"

namespace dl {

namespace optim { class Optimizer; }

DL_API bool is_autocast_enabled(DeviceType device);
DL_API DType autocast_dtype(DeviceType device);

/// 当前线程中为选定设备类型启用低精度 GEMM 与卷积；dtype 为 f16/bf16，作用域可以嵌套。
/// 主参数、梯度、优化器与低精度计算的输出保持 FP32，前向记录反向所用的精度。
class DL_API AutocastGuard {
public:
    explicit AutocastGuard(DeviceType device = DeviceType::cuda, DType dtype = DType::bf16, bool enabled = true);
    ~AutocastGuard();
    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;
private:
    DeviceType device_;
    DType previous_dtype_;
    bool previous_enabled_;
};

struct GradScalerOptions {
    /// 倍率使用 FP32 正常正数范围；增长因子大于一，回退因子在 (0, 1) 内。
    double initial_scale = 65536.0;
    double growth_factor = 2.0;
    double backoff_factor = 0.5;
    /// 连续有限梯度的更新窗口达到此正整数后，增加倍率。
    Index growth_interval = 2000;
    bool enabled = true;
};

struct GradScalerState {
    GradScalerOptions options;
    /// 以 double 字段保存精确的 FP32 倍率值，finite_steps 位于 [0, growth_interval)。
    double scale = 65536.0;
    Index finite_steps = 0;
};

struct UnscaledGradients {
    Gradients gradients;
    bool finite;
};

/// 动态损失缩放与 autocast 分别管理梯度范围和前向精度。
/// 同一更新窗口的所有 backward 使用相同倍率，之后还原梯度、更新参数并调整倍率。
class DL_API GradScaler {
public:
    explicit GradScaler(GradScalerOptions options = {});
    bool enabled() const { return state_.options.enabled; }
    double get_scale() const { return enabled() ? state_.scale : 1.0; }
    Tensor scale(const Tensor& loss) const;

    /// 每个更新窗口调用一次，原地还原优化器已有梯度；返回这些梯度是否全部有限。
    /// CPU 梯度可为 f32/f64，CUDA 为 f32；CUDA 每个设备只回传一个合并标志。
    /// 需要裁剪时，在本调用之后执行，再按返回值决定是否调用 optimizer.step。
    bool unscale(optim::Optimizer& optimizer) const;
    /// 独立返回还原后的梯度与合并有限标志，空项和重复身份对应关系保持原样。
    /// 每个有值身份复制为连续、无求导历史的存储，原梯度只读，状态倍率保持不变。
    UnscaledGradients unscale(std::span<const std::optional<Tensor>> gradients) const;
    /// 原地处理连续可写梯度，同身份只处理一次，不同身份的逻辑元素彼此不重叠。
    /// CPU 为 f32/f64，CUDA 为 f32；多个设备分别还原后合并标志，空列表返回 true。
    /// 禁用时不扫描或修改数值并返回 true；独立返回入口仍复制结果。
    bool unscale_(std::span<const std::optional<Tensor>> gradients) const;
    /// 调整下一窗口的倍率：非有限时回退并清零计数，连续有限时按间隔增长。
    void update(bool finite);

    /// 组合 unscale、条件 optimizer.step 和 update；返回是否调用了优化器更新。
    /// 此入口用于尚未还原梯度的窗口；显式还原或裁剪的流程使用上面的两个入口。
    bool step(optim::Optimizer& optimizer);

    /// 在 update 或 step 完成的窗口边界保存状态，恢复后从下一窗口开始累计。
    GradScalerState state_dict() const { return state_; }
    void load_state_dict(const GradScalerState& state) { state_ = state; }

private:
    GradScalerState state_;
};

} // namespace dl
