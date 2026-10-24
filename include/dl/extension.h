#pragma once

#include "dl/autograd.h"
#include "dl/tensor.h"

#include <functional>
#include <initializer_list>

namespace dl {

/// 独立于数据存储的张量规格；stride 和 storage_offset 均以元素为单位。
struct DL_API TensorSpec {
    Shape shape, strides;
    DType dtype = DType::f32;
    Device device = Device::cpu();
    Index storage_offset = 0;
    bool read_only = false;

    Index numel() const;
    bool is_contiguous() const;
};

DL_API TensorSpec tensor_spec(const Tensor& tensor);
DL_API TensorSpec contiguous_spec(Shape shape, DType dtype = DType::f32, Device device = Device::cpu());

/// 每次自定义调用拥有独立上下文，输入规格与求导需求在进入无图前向前记录。
class DL_API VjpContext {
public:
    explicit VjpContext(std::span<const Tensor> inputs);
    std::size_t input_count() const { return inputs_.size(); }
    const TensorSpec& input_spec(std::size_t index) const { return inputs_[index]; }
    bool needs_input_grad(std::size_t index) const { return needs_[index]; }
    bool needs_grad() const { return recording_; }

    /// 替换保存值列表，张量共享存储并脱离求导历史；无需求导时不保留数据。
    void save_for_backward(std::span<const Tensor> tensors);
    void save_for_backward(std::initializer_list<Tensor> tensors) {
        save_for_backward(std::span<const Tensor>(tensors.begin(), tensors.size()));
    }
    std::span<const Tensor> saved_tensors() const { return saved_; }

private:
    std::vector<TensorSpec> inputs_;
    std::vector<bool> needs_;
    std::vector<Tensor> saved_;
    bool recording_ = false;
};

using VjpGradients = Gradients;
using VjpForward = std::function<Tensor(VjpContext&, std::span<const Tensor>)>;
using VjpBackward = std::function<VjpGradients(const VjpContext&, const Tensor&)>;

/// 前向与反向在 NoGrad 下执行；反向恢复本次前向入口的 CPU/CUDA autocast 设置。
/// backward 接收输出累积 dtype 的种子，当前接口表达单输出的一阶反向规则。
/// forward 只读输入并返回一个张量；backward 按输入顺序返回同形、同设备的梯度或 nullopt。
/// 梯度使用输入的累积 dtype，广播归约由规则自身完成；重复输入的贡献分别累计。
/// 保存的张量通过 context 管理，非张量配置可由 backward 闭包按值持有。
/// 输出共享 forward 结果的存储，原求导历史由本次 VJP 替换；非浮点输出不参与求导。
DL_API Tensor custom_vjp(std::span<const Tensor> inputs, const VjpForward& forward, VjpBackward backward);
inline Tensor custom_vjp(std::initializer_list<Tensor> inputs, const VjpForward& forward, VjpBackward backward) {
    return custom_vjp(std::span<const Tensor>(inputs.begin(), inputs.size()), forward, std::move(backward));
}

} // namespace dl
