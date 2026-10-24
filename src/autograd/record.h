#pragma once

#include "core/tensor_impl.h"
#include "dl/autograd.h"

#include <algorithm>

namespace dl::detail {

inline Edge gradient_edge(const Tensor& tensor) {
    const auto* meta = TensorAccess::autograd(tensor);
    return is_grad_enabled() && meta ? meta->edge : Edge{};
}

/// 每个闭包的保存值具有具体类型，节点通过一次虚调用执行整项 VJP。
template <class Function> class BackwardNode final : public Node {
public:
    BackwardNode(std::vector<Edge> inputs, Function function)
        : Node(std::move(inputs)), function_(std::move(function)) {}

    void backward(TensorData grad, GradSink& sink) override {
        // 中间梯度也可被 grad 的结果列表持有，VJP 通过只读借用计算各输入贡献。
        (*function_)(grad, sink, inputs);
    }
    void release_saved() override { function_.reset(); }

private:
    std::optional<Function> function_;
};

template <class Function>
Tensor record(TensorData data, std::vector<Edge> inputs, Function function) {
    const bool tracks_gradient = std::any_of(inputs.begin(), inputs.end(),
                                            [](const Edge& edge) { return bool(edge.node); });
    if (!tracks_gradient) {
        return make_tensor(std::move(data));
    }
    auto node = std::make_shared<BackwardNode<Function>>(std::move(inputs), std::move(function));
    auto meta = std::make_unique<AutogradMeta>(AutogradMeta{{std::move(node)}, {}});
    return TensorAccess::wrap(std::move(data), std::move(meta));
}

} // namespace dl::detail
