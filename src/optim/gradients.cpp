#include "optim/gradients.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <unordered_map>
#include <unordered_set>

namespace dl::detail {

std::vector<Tensor> leaf_gradients(std::span<const Tensor> parameters) {
    std::unordered_set<const void*> seen;
    std::vector<Tensor> result;
    result.reserve(parameters.size());
    for (const auto& parameter : parameters) {
        if (seen.insert(TensorAccess::identity(parameter)).second) {
            if (auto gradient = parameter.grad()) result.push_back(std::move(*gradient));
        }
    }
    return result;
}

std::vector<Tensor> unique_gradients(std::span<const std::optional<Tensor>> gradients) {
    std::unordered_set<const void*> seen;
    std::vector<Tensor> result;
    result.reserve(gradients.size());
    for (const auto& gradient : gradients) {
        if (gradient && seen.insert(TensorAccess::identity(*gradient)).second)
            result.push_back(*gradient);
    }
    return result;
}

Gradients clone_gradients(std::span<const std::optional<Tensor>> gradients) {
    std::unordered_map<const void*, std::size_t> first_occurrence;
    Gradients result(gradients.size());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        if (const auto& gradient = gradients[i]) {
            const auto [entry, inserted] = first_occurrence.try_emplace(TensorAccess::identity(*gradient), i);
            result[i] = inserted ? make_tensor(numeric::clone(data_of(*gradient))) : *result[entry->second];
        }
    }
    return result;
}

} // namespace dl::detail
