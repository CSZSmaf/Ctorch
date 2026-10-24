#pragma once

#include "dl/nn.h"
#include "dl/random.h"

#include <functional>

namespace dl::nn {
namespace functional_detail {
struct FunctionalLayout;
struct FunctionalContext;
} // namespace functional_detail

/// 槽顺序分别对应 FunctionalModule 的参数、buffer 和模块名称，模式以 0/1 表示 eval/train。
/// 参数保留求导身份，buffer 作为无求导状态传递，随机 key 由每次调用单独提供。
struct FunctionalState {
    std::vector<Tensor> parameters;
    std::vector<Tensor> buffers;
    std::vector<std::uint8_t> training;
};

template <class Output>
struct FunctionalResult {
    Output output;
    std::vector<Tensor> buffers;
};

/// 固定模块树的参数绑定表，名称和共享身份在构造时整理一次。
/// 原模块树在此对象及其调用期间保持存活，登记结构和参数身份保持稳定。
/// 内置层通过 bound/bound_buffer 读取外部状态；普通 forward 继续读取自身登记值。
/// 嵌套时每个模块选择最近的所属绑定，树外模块保留普通调用语义。
class DL_API FunctionalModule {
public:
    explicit FunctionalModule(Module& module);
    std::span<const std::string> parameter_names() const;
    std::span<const std::string> buffer_names() const;
    std::span<const std::string> module_names() const;

    /// 从原模块当前数值建立独立状态，参数成为同求导开关的新叶，buffer 脱离求导历史。
    FunctionalState state() const;

    /// forward 在当前线程同步执行，可返回 Tensor、RecurrentResult 或其他张量值结构。
    /// state 的槽数量、顺序及张量规格对应模块定义；参数只读，buffer 先复制再更新。
    /// 前向及其子模块通过 bound/bound_buffer、is_training 和 next_random_key 读取本次绑定。
    /// callback 可组合其他模块；它们按自身所属的外层绑定或原登记状态执行。
    template <class Forward>
    auto call(const FunctionalState& state, RandomKey key, Forward&& forward) const {
        return call(state.parameters, state.buffers, state.training, key, std::forward<Forward>(forward));
    }

    /// 借用连续的状态槽，可直接接收图输入的子区间；仅 buffer 数据建立本次独立副本。
    /// 三个 span 只需在同步前向期间存活，返回值与求导保存值继续持有实际使用的张量。
    template <class Forward>
    auto call(std::span<const Tensor> parameters, std::span<const Tensor> buffers,
              std::span<const std::uint8_t> training, RandomKey key, Forward&& forward) const {
        Invocation invocation(*this, parameters, buffers, training, key);
        auto output = std::invoke(std::forward<Forward>(forward));
        return FunctionalResult<decltype(output)>{std::move(output), invocation.take_buffers()};
    }

private:
    class DL_API Invocation {
    public:
        Invocation(const FunctionalModule& module, std::span<const Tensor> parameters,
                   std::span<const Tensor> buffers, std::span<const std::uint8_t> training, RandomKey key);
        ~Invocation();
        std::vector<Tensor> take_buffers();

    private:
        std::unique_ptr<functional_detail::FunctionalContext> context_;
    };
    std::shared_ptr<const functional_detail::FunctionalLayout> layout_;
};

} // namespace dl::nn
