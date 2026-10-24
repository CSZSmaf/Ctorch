#include "dl/extension.h"

#include "autograd/record.h"
#include "core/amp.h"

namespace dl {

VjpContext::VjpContext(std::span<const Tensor> inputs) {
    inputs_.reserve(inputs.size());
    needs_.reserve(inputs.size());
    for (const auto& input : inputs) {
        inputs_.push_back(tensor_spec(input));
        const bool needed = is_grad_enabled() && input.requires_grad();
        needs_.push_back(needed);
        recording_ |= needed;
    }
}

void VjpContext::save_for_backward(std::span<const Tensor> tensors) {
    std::vector<Tensor> values;
    if (recording_) {
        values.reserve(tensors.size());
        for (const auto& tensor : tensors) values.push_back(tensor.detach());
    }
    saved_ = std::move(values);
}

Tensor custom_vjp(std::span<const Tensor> inputs, const VjpForward& forward, VjpBackward backward) {
    VjpContext context(inputs);
    std::vector<detail::Edge> edges;
    edges.reserve(inputs.size());
    for (const auto& input : inputs) edges.push_back(detail::gradient_edge(input));

    const detail::AutocastState precision;
    auto output = [&] {
        NoGradGuard guard;
        return forward(context, inputs);
    }();
    if (!context.needs_grad() || !is_floating(output.dtype())) return output.detach();

    // 输出建立新身份，别名仍由 TensorData 表达；上下文保存值不持有输入或输出的求导节点。
    return detail::record(detail::data_of(output), std::move(edges),
        [context = std::move(context), backward = std::move(backward), precision]
        (const auto& grad, auto& sink, const auto& input_edges) {
            NoGradGuard guard;
            detail::AutocastReplay autocast(precision);
            auto contributions = backward(context, detail::make_tensor(grad));
            for (std::size_t index = 0; index < input_edges.size(); ++index) {
                if (sink.needs(input_edges[index]) && contributions[index])
                    sink.add(input_edges[index], detail::data_of(*contributions[index]));
            }
        });
}

} // namespace dl
