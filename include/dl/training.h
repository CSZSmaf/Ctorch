#pragma once

#include "dl/nn.h"
#include "dl/optim.h"

namespace dl::training {

enum class MetricMode { minimize, maximize };
struct MonitorOptions {
    MetricMode mode = MetricMode::minimize;
    Index patience = 10;
    double min_delta = 0.0;
};
struct MetricHistory { double best = 0.0; Index bad_steps = 0; bool initialized = false; };

/// 每轮验证后传入有限指标；变化超过 min_delta 才计为改善，patience 为正。
/// step 返回是否达到连续未改善次数；最佳权重可由调用方另存为检查点。
class DL_API EarlyStopping {
public:
    explicit EarlyStopping(MonitorOptions options = {}) : options_(options) {}
    bool step(double metric);
    double best() const { return history_.best; }
    StateDict state_dict() const;
    void load_state_dict(std::span<const NamedTensor> state);
private:
    MonitorOptions options_;
    MetricHistory history_;
};

struct PlateauOptions {
    MonitorOptions monitor;
    double factor = 0.1;
    double minimum_rate = 0.0;
};

/// 验证指标连续超过 patience 次未改善时，按 factor 缩小各组当前学习率。
/// patience 可为零，factor 在 (0,1) 内；优化器的生命周期覆盖本对象。
class DL_API ReduceLROnPlateau {
public:
    ReduceLROnPlateau(optim::Optimizer& optimizer, PlateauOptions options = {});
    bool step(double metric);
    StateDict state_dict() const;
    /// 恢复监控状态和配置；当前学习率随优化器状态恢复。
    void load_state_dict(std::span<const NamedTensor> state);
private:
    optim::Optimizer& optimizer_;
    PlateauOptions options_;
    MetricHistory history_;
};

/// 参数采用 decay * average + (1-decay) * parameter，buffer 复制本次模型值。
/// 在 optimizer.step 后更新；浮点参数以累积精度保存，copy_to 写入同结构模型。
/// state_dict 与内部平均值共享存储，clone_state 或检查点保存可取得独立快照。
class DL_API ExponentialMovingAverage {
public:
    explicit ExponentialMovingAverage(const nn::Module& model, double decay = 0.999);
    void update(const nn::Module& model);
    void copy_to(nn::Module& model) const;
    Index steps() const { return steps_; }
    StateDict state_dict() const;
    void load_state_dict(std::span<const NamedTensor> state);
private:
    StateDict averages_;
    std::vector<bool> parameters_;
    double decay_;
    Index steps_ = 0;
};

struct ClassificationScores {
    double accuracy = 0.0;
    std::vector<double> precision, recall, f1, iou;
};

/// 行为真实类别，列为预测类别；类别数为正，输入是同形状的 int64 编号，范围为 [0,classes)。
/// update 将编号复制到 CPU 后累计整数计数，适合验证阶段；ignore_index 跳过对应目标。
class DL_API ConfusionMatrix {
public:
    explicit ConfusionMatrix(Index classes);
    void update(const Tensor& prediction, const Tensor& target, std::optional<Index> ignore_index = std::nullopt);
    void reset();
    /// 无样本的比率取零；返回逐类别指标，调用方可按任务选择平均方法。
    ClassificationScores scores() const;
    Tensor state_dict() const;
    void load_state_dict(const Tensor& counts);
private:
    Index classes_;
    std::vector<Index> counts_;
};

} // namespace dl::training
