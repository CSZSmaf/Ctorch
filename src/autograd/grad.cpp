#include "dl/autograd.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <unordered_map>

namespace dl {
namespace {

detail::Edge existing_edge(const Tensor& tensor) {
    const auto* meta = detail::TensorAccess::autograd(tensor);
    return meta ? meta->edge : detail::Edge{};
}

} // namespace

Gradients grad(std::span<const Tensor> outputs, std::span<const Tensor> inputs,
               std::span<const Tensor> seeds) {
    std::vector<detail::Edge> roots, requests;
    std::vector<detail::TensorData> cotangents;
    roots.reserve(outputs.size());
    cotangents.reserve(outputs.size());
    requests.reserve(inputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (auto edge = existing_edge(outputs[i]); edge.node) {
            roots.push_back(std::move(edge));
            cotangents.push_back(seeds.empty()
                ? detail::numeric::full({}, 1.0, accumulation_dtype(outputs[i].dtype()), outputs[i].device())
                : detail::data_of(seeds[i]));
        }
    }
    // 记录模式只决定前向是否建图；已有图在 NoGrad 作用域内也可求取梯度。
    for (const auto& input : inputs) requests.push_back(existing_edge(input));
    auto values = detail::run_grad(roots, requests, cotangents);
    Gradients result(inputs.size());
    std::unordered_map<const detail::Node*, std::size_t> first_occurrence;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i]) {
            const auto [entry, inserted] = first_occurrence.try_emplace(requests[i].node.get(), i);
            // 同一请求共享结果身份，后续按身份处理梯度时只还原或裁剪一次。
            result[i] = inserted ? detail::make_tensor(std::move(*values[i])) : *result[entry->second];
        }
    }
    return result;
}

Gradients grad(const Tensor& output, std::span<const Tensor> inputs,
               const std::optional<Tensor>& seed) {
    return grad(std::span<const Tensor>(&output, 1), inputs,
                seed ? std::span<const Tensor>(&*seed, 1) : std::span<const Tensor>{});
}

} // namespace dl
