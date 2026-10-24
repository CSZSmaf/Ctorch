#pragma once

#include "dl/tensor.h"

#include <functional>
#include <initializer_list>
#include <utility>

namespace dl {

DL_API bool is_grad_enabled();

/// 显式选择当前线程的记录模式，离开作用域时恢复原模式；可在 NoGrad 内临时开启记录。
class DL_API GradModeGuard {
public:
    explicit GradModeGuard(bool enabled);
    ~GradModeGuard();
    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool previous_;
};

/// 在当前线程中暂停记录运算，离开作用域时恢复进入前的记录状态。
class DL_API NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    GradModeGuard mode_;
};

struct CheckpointOptions {
    /// 保存当前线程默认 Generator 的位置，重算后恢复进入反向时的位置。
    /// 显式 RandomKey 或无随机计算的函数可关闭此项。
    bool preserve_rng_state = true;
};
using CheckpointFunction = std::function<Tensor(std::span<const Tensor>)>;

/// 前向只保存显式输入，反向重算整个函数并用已有一阶引擎交付梯度。
/// 所有参与求导的参数也放入 inputs；function 只读输入，闭包持有固定配置。
/// 数学结果由输入值、随机序列和固定配置决定，记录模式仅影响求导关系。
/// 函数按相同输入与随机序列重现结果，模块 buffer 等外部状态由调用方在块外更新。
/// 重算恢复前向的 CPU/CUDA autocast；默认随机流按 options 处理，显式 Generator 由函数管理。
/// 单个浮点输出连接新节点，视图保留实际存储；无需求导时直接执行 function。
DL_API Tensor checkpoint(std::span<const Tensor> inputs, CheckpointFunction function,
                         CheckpointOptions options = {});
inline Tensor checkpoint(std::initializer_list<Tensor> inputs, CheckpointFunction function,
                         CheckpointOptions options = {}) {
    return checkpoint(std::span<const Tensor>(inputs.begin(), inputs.size()), std::move(function), options);
}

/// 每项对应一个请求的输入；没有收到反向贡献的输入保留空值。
using Gradients = std::vector<std::optional<Tensor>>;

/// 返回各输出 VJP 的和，按 inputs 顺序交付，不读取或写入任何叶梯度槽。
/// seeds 为空时 outputs 均为标量，分别使用种子 1；否则数量、shape、设备对应各输出，
/// dtype 使用输出 dtype 或其累积 dtype。输入可为叶或中间结果，冻结和未使用项返回空值。
/// 一次调用消费 outputs 的可达前向图，包括未参与数值反向的分支；返回值不建立求导图。
/// 重复输入返回同一梯度身份，重复输出的种子分别参与累计。
DL_API Gradients grad(std::span<const Tensor> outputs, std::span<const Tensor> inputs,
                      std::span<const Tensor> seeds = {});

/// 单输出入口在未提供 seed 时对标量使用种子 1，其他约定与多输出入口一致。
DL_API Gradients grad(const Tensor& output, std::span<const Tensor> inputs,
                      const std::optional<Tensor>& seed = std::nullopt);

inline Gradients grad(const Tensor& output, std::initializer_list<Tensor> inputs,
                      const std::optional<Tensor>& seed = std::nullopt) {
    return grad(output, std::span<const Tensor>(inputs.begin(), inputs.size()), seed);
}

inline Gradients grad(std::initializer_list<Tensor> outputs, std::initializer_list<Tensor> inputs,
                      std::initializer_list<Tensor> seeds = {}) {
    return grad(std::span<const Tensor>(outputs.begin(), outputs.size()),
                std::span<const Tensor>(inputs.begin(), inputs.size()),
                std::span<const Tensor>(seeds.begin(), seeds.size()));
}

} // namespace dl
