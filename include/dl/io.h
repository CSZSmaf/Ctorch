#pragma once

#include "dl/amp.h"
#include "dl/data.h"
#include "dl/functional.h"
#include "dl/nn.h"
#include "dl/optim.h"
#include "dl/scheduler.h"

#include <filesystem>
#include <functional>

namespace dl::io {

/// 保存名称、dtype、shape 与连续张量值；模型结构由调用方的 C++ 类型表达。
DL_API void save(const std::filesystem::path& path, std::span<const NamedTensor> state);
DL_API StateDict load(const std::filesystem::path& path);

struct TrainingProgress {
    /// epoch 标记当前轮次，step 记录已经完成的优化器更新数。
    Index epoch = 0;
    Index step = 0;
};

struct NamedGenerator {
    std::string name;
    std::reference_wrapper<Generator> generator;
};

struct NamedScheduler {
    std::string name;
    std::reference_wrapper<optim::LRScheduler> scheduler;
};

/// 采集得到的模型与优化器张量是独立 CPU 副本；模型结构、数据集及生成器对象由调用方构造。
struct TrainingCheckpoint {
    StateDict model;
    nn::TrainingModes modes;
    std::vector<std::string> optimizer_parameters;
    optim::OptimizerState optimizer;
    TrainingProgress progress;
    std::string random_state;
    std::optional<data::LoaderState> loader;
    std::vector<std::pair<std::string, std::string>> generators;
    std::vector<std::pair<std::string, optim::SchedulerState>> schedulers;
    /// 应用层的命名张量，例如当前轮次的损失累计和样本数。
    StateDict extra;
    std::optional<data::PrefetchState> prefetch;
    std::optional<GradScalerState> scaler;
    /// 逐参数求导状态独立于 train/eval；旧版本快照缺省时沿用调用方的设置。
    std::optional<nn::GradientModes> gradient_modes;
    /// 显式随机流的 root 或下一次调用 key；v6 以完整的两个 uint64 字段保存。
    std::optional<RandomKey> random_key;
};

/// 以非空命名空间保存训练工具状态。set_extra 复制独立 CPU 值，替换该名称下的旧状态。
DL_API void set_extra(TrainingCheckpoint& checkpoint, std::string_view name, std::span<const NamedTensor> state);
DL_API StateDict get_extra(const TrainingCheckpoint& checkpoint, std::string_view name);

/// 在一次更新完成后采集状态；参数按模型的规范名称关联，默认随机流一并保存。
DL_API TrainingCheckpoint capture_checkpoint(const nn::Module& model, const optim::Optimizer& optimizer,
    TrainingProgress progress = {}, const data::DataLoader* loader = nullptr,
    std::span<const NamedGenerator> generators = {}, std::span<const NamedScheduler> schedulers = {},
    const GradScaler* scaler = nullptr);

/// 恢复相同模型、算法和数据集的状态，并清空叶梯度。
/// 模型先迁移到目标设备，再构造优化器；参数集合相同，列表顺序可以不同。
/// 带采样器的快照对应一个 loader，各额外生成器由保存时的名称关联。
/// 各调度器先绑定当前优化器，再按名称恢复；参数组及学习率一起恢复。
/// 传入 scaler 时恢复快照中的缩放状态；旧快照没有此状态时保留调用方的初始配置。
/// v5 同时恢复参数冻结状态；v1-v4 保留调用方的 requires_grad 选择。
DL_API TrainingProgress restore_checkpoint(const TrainingCheckpoint& checkpoint, nn::Module& model,
    optim::Optimizer& optimizer, data::DataLoader* loader = nullptr,
    std::span<const NamedGenerator> generators = {}, std::span<const NamedScheduler> schedulers = {},
    GradScaler* scaler = nullptr);

/// 预取入口同时保存生产端采样位置与尚未消费的批次，恢复后保持相同的数据和随机流顺序。
DL_API TrainingCheckpoint capture_checkpoint(const nn::Module& model, const optim::Optimizer& optimizer,
    TrainingProgress progress, const data::PrefetchLoader& loader,
    std::span<const NamedGenerator> generators = {}, std::span<const NamedScheduler> schedulers = {},
    const GradScaler* scaler = nullptr);
DL_API TrainingProgress restore_checkpoint(const TrainingCheckpoint& checkpoint, nn::Module& model,
    optim::Optimizer& optimizer, data::PrefetchLoader& loader,
    std::span<const NamedGenerator> generators = {}, std::span<const NamedScheduler> schedulers = {},
    GradScaler* scaler = nullptr);

/// 函数式训练恢复交付新叶、独立 buffer 和优化器状态，原 Module 保留自己的登记值。
struct FunctionalTrainingState {
    nn::FunctionalState state;
    optim::OptimizerState optimizer;
    RandomKey random_key;
    TrainingProgress progress;
};

/// 优化器槽按 functional.parameter_names() 的完整顺序提供，包含冻结或尚无动量的参数。
/// 捕获外部参数/模式而非原 Module，张量和优化器状态保存为独立 CPU 副本。
DL_API TrainingCheckpoint capture_checkpoint(const nn::FunctionalModule& functional,
    const nn::FunctionalState& state, const optim::OptimizerState& optimizer, RandomKey random_key,
    TrainingProgress progress = {}, const data::DataLoader* loader = nullptr,
    std::span<const NamedGenerator> generators = {}, const GradScaler* scaler = nullptr);
/// 恢复同一组规范名称；快照包含显式 key 与逐参数求导模式，返回值使用指定目标设备。
/// DataLoader、额外 Generator 与默认随机流沿用普通训练快照的恢复约定。
DL_API FunctionalTrainingState restore_checkpoint(const TrainingCheckpoint& checkpoint,
    const nn::FunctionalModule& functional, Device device = Device::cpu(),
    data::DataLoader* loader = nullptr, std::span<const NamedGenerator> generators = {},
    GradScaler* scaler = nullptr);

DL_API TrainingCheckpoint capture_checkpoint(const nn::FunctionalModule& functional,
    const nn::FunctionalState& state, const optim::OptimizerState& optimizer, RandomKey random_key,
    TrainingProgress progress, const data::PrefetchLoader& loader,
    std::span<const NamedGenerator> generators = {}, const GradScaler* scaler = nullptr);
DL_API FunctionalTrainingState restore_checkpoint(const TrainingCheckpoint& checkpoint,
    const nn::FunctionalModule& functional, Device device, data::PrefetchLoader& loader,
    std::span<const NamedGenerator> generators = {}, GradScaler* scaler = nullptr);

DL_API void save_checkpoint(const std::filesystem::path& path, const TrainingCheckpoint& checkpoint);
DL_API TrainingCheckpoint load_checkpoint(const std::filesystem::path& path);

} // namespace dl::io
