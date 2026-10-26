#pragma once

#include "dl/random.h"

#include <filesystem>
#include <functional>

namespace dl::data {

struct TensorDataset {
    Tensor input;
    Tensor target;
    Index size() const { return input.size(0); }
};

struct Batch {
    Tensor input;
    Tensor target;
    /// 依次保存 mask、长度或其他模型输入，随批次一起传输和保存。
    std::vector<Tensor> extra = {};
};

/// 按给定样本编号整理一个批次；函数拥有或借用的数据集在 DataLoader 使用期间有效。
/// 每次调用可读取文件、填充变长样本或组合张量，返回的 Batch 拥有本批数据。
using Collate = std::function<Batch(std::span<const Index>)>;

struct LoaderOptions {
    /// 每批样本数为正；末批可以保留剩余样本，或由 drop_last 跳过。
    Index batch_size = 64;
    bool shuffle = true;
    bool drop_last = false;
};

/// order 保存当前轮次的排列，cursor 指向下一批的首个样本。
struct LoaderState {
    LoaderOptions options;
    std::vector<Index> order;
    Index cursor = 0;
    std::string random_state;
};

/// 按样本编号迭代数据集，reset 开始新一轮采样；TensorDataset 按第一维组织样本。
class DL_API DataLoader {
public:
    DataLoader(TensorDataset dataset, LoaderOptions options = {},
               Generator& generator = default_generator());
    DataLoader(Index size, Collate collate, LoaderOptions options = {},
               Generator& generator = default_generator());
    void reset();
    /// 按指定样本编号开始一轮；可重复采样，也可只遍历数据集的一部分。
    void reset(std::vector<Index> order);
    /// 批次保留整理函数返回的存储与设备；TensorDataset 入口生成独立副本，构造后游标前移。
    std::optional<Batch> next();
    LoaderState state_dict() const;
    /// 恢复同一数据集的排列、位置和采样随机流；下一次 next 延续当前轮次。
    void load_state_dict(const LoaderState& state);

private:
    Index size_;
    Collate collate_;
    LoaderOptions options_;
    Generator& generator_;
    std::vector<Index> order_;
    Index cursor_ = 0;
};

/// 已整理但尚未交付的批次随预取状态保存，恢复时无需重新执行其数据增强。
struct PrefetchState {
    std::size_t depth = 2;
    std::vector<Batch> pending;
    bool background = false;
};

/// 在目标设备预取有限批次，depth 为正，源 loader 的生命周期覆盖本对象。
/// background 开启单个 CPU 整理线程；数据增强使用独立 Generator，设备传输在调用线程提交。
/// reset、next 和状态恢复统一通过本对象调用；运行期间 source 由预取器独占。
/// 复用只写入预取器自建的暂存副本，整理函数交付的页锁定存储按传输规则借用。
class DL_API PrefetchLoader {
public:
    PrefetchLoader(DataLoader& source, Device device, std::size_t depth = 2, bool background = false);
    ~PrefetchLoader();
    PrefetchLoader(const PrefetchLoader&) = delete;
    PrefetchLoader& operator=(const PrefetchLoader&) = delete;
    std::optional<Batch> next();
    void reset();
    void reset(std::vector<Index> order);
    DataLoader& source() const;
    /// 等待当前整理结束，保存待交付值的独立 CPU 副本；下一次 next 重启后台整理。
    /// 随后读取 source 的 LoaderState 与增强 Generator 状态，可得到一致的检查点。
    PrefetchState state_dict() const;
    void load_state_dict(const PrefetchState& state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct PaddedSequence {
    Tensor values;  // [batch, max_length, ...]，右侧填充。
    Tensor lengths; // int64[batch]。
    Tensor mask;    // bool[batch, max_length]，有效位置为 true。
};

/// 序列列表非空，各项的 dtype、设备和除首维外的形状相同；支持长度为零的序列。
DL_API PaddedSequence pad_sequence(std::span<const Tensor> sequences, double value = 0.0);
/// 非负权重中至少一项为正；有放回采样 count 个编号。
DL_API std::vector<Index> weighted_order(std::span<const double> weights, Index count,
    Generator& generator = default_generator());
/// 先打乱样本，再在窗口内按长度分组，最后打乱批次顺序，减少填充并保留随机性。
/// batch_size 与 bucket_batches 为正；不足一批的样本放在末尾。
DL_API std::vector<Index> bucket_order(std::span<const Index> lengths, Index batch_size,
    Generator& generator = default_generator(), Index bucket_batches = 32);

/// 读取已解压的 MNIST IDX 图像和标签，返回 [N,1,H,W] 的 float32 图像与 int64 标签。
/// 像素按 255 归一化到 [0,1]，两个文件按原有样本顺序对应。
DL_API TensorDataset load_mnist(const std::filesystem::path& images, const std::filesystem::path& labels);

} // namespace dl::data
