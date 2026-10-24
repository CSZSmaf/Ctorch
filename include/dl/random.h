#pragma once

#include "dl/tensor.h"

#include <array>
#include <random>
#include <string>
#include <string_view>

namespace dl {

/// 显式 Philox 随机流；采样按值读取 key，同 key 和相同参数重复得到同一序列。
/// seed 选择轮密钥，stream 选择 counter 的高 64 位，元素编号占低 64 位。
struct RandomKey {
    std::uint64_t seed, stream = 0;
    bool operator==(const RandomKey&) const = default;
};

/// 根据标签派生新 key，原 key 保持不变；split 按从零开始的标签顺序分流。
DL_API RandomKey fold_in(RandomKey key, std::uint64_t label) noexcept;
DL_API std::array<RandomKey, 2> split(RandomKey key) noexcept;
DL_API std::vector<RandomKey> split(RandomKey key, std::size_t count);

/// 将 key 的两段位模式存入 CPU int64[2]，可作为固定计划输入或普通状态张量保存。
/// 读取入口接受 CPU 上连续的 int64[2]，不进行设备传输。
DL_API Tensor key_tensor(RandomKey key);
DL_API RandomKey key_from_tensor(const Tensor& tensor);

/// 随机序列由生成器拥有；共享同一个生成器可按调用顺序重现初始化与采样。
class DL_API Generator {
public:
    explicit Generator(std::uint64_t seed = 0) : engine_(seed) {}
    void seed(std::uint64_t value) { engine_.seed(value); }
    std::mt19937_64& engine() { return engine_; }
    void shuffle(std::span<Index> values);

    /// 状态文本记录引擎的当前位置，恢复后继续消费同一随机流。
    std::string state() const;
    void load_state(std::string_view state);

private:
    std::mt19937_64 engine_;
};

DL_API Generator& default_generator();
DL_API void manual_seed(std::uint64_t seed);
/// CPU 使用标准库分布；CUDA 每次非空调用从生成器取得一个 key，由 Philox 在设备上展开。
/// Generator 状态包含后续 key 序列；同版本、同后端、同调用顺序可恢复随机流。
DL_API Tensor uniform(Shape shape, double low, double high, TensorOptions options = {},
               Generator& generator = default_generator());
DL_API Tensor randn(Shape shape, TensorOptions options = {}, Generator& generator = default_generator());
DL_API Tensor bernoulli(Shape shape, double probability, Generator& generator = default_generator(),
                 Device device = Device::cpu());

/// 显式 key 不消费 Generator；CPU/CUDA 共用整数展开，浮点分布遵循各后端计算精度。
DL_API Tensor uniform(RandomKey key, Shape shape, double low, double high, TensorOptions options = {});
DL_API Tensor randn(RandomKey key, Shape shape, TensorOptions options = {});
DL_API Tensor bernoulli(RandomKey key, Shape shape, double probability, Device device = Device::cpu());

} // namespace dl
