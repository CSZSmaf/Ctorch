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
};

/// 在目标设备预取有限批次，depth 为正，源 loader 的生命周期覆盖本对象。
/// 整理函数在调用线程执行，页锁定缓冲复用；reset 和 next 统一通过本对象调用。
/// 复用只写入预取器自建的暂存副本，整理函数交付的页锁定存储按传输规则借用。
class DL_API PrefetchLoader {
public:
    PrefetchLoader(DataLoader& source, Device device, std::size_t depth = 2);
    ~PrefetchLoader();
    PrefetchLoader(const PrefetchLoader&) = delete;
    PrefetchLoader& operator=(const PrefetchLoader&) = delete;
    std::optional<Batch> next();
    void reset();
    DataLoader& source() const;
    /// 状态包含实际待交付值的独立 CPU 副本；采样位置由 source 的 LoaderState 一同保存。
    PrefetchState state_dict() const;
    void load_state_dict(const PrefetchState& state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// 读取已解压的 MNIST IDX 图像和标签，返回 [N,1,H,W] 的 float32 图像与 int64 标签。
/// 像素按 255 归一化到 [0,1]，两个文件按原有样本顺序对应。
DL_API TensorDataset load_mnist(const std::filesystem::path& images, const std::filesystem::path& labels);

} // namespace dl::data
