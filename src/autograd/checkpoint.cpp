#include "dl/autograd.h"

#include "autograd/record.h"
#include "core/amp.h"
#include "dl/random.h"

#include <algorithm>
#include <utility>

namespace dl {
namespace {

// 保存值属于一次性求导图。交换引擎状态后重算，结束时将调用线程原来的位置换回。
class RandomReplay {
public:
    explicit RandomReplay(std::mt19937_64* saved) : saved_(saved) {
        if (saved_) std::swap(default_generator().engine(), *saved_);
    }
    ~RandomReplay() { if (saved_) std::swap(default_generator().engine(), *saved_); }
    RandomReplay(const RandomReplay&) = delete;
    RandomReplay& operator=(const RandomReplay&) = delete;

private:
    std::mt19937_64* saved_;
};

} // namespace

Tensor checkpoint(std::span<const Tensor> inputs, CheckpointFunction function, CheckpointOptions options) {
    std::vector<detail::Edge> edges;
    edges.reserve(inputs.size());
    for (const auto& input : inputs) edges.push_back(detail::gradient_edge(input));
    if (std::none_of(edges.begin(), edges.end(), [](const auto& edge) { return bool(edge.node); }))
        return function(inputs);

    std::vector<detail::TensorData> saved;
    saved.reserve(inputs.size());
    for (const auto& input : inputs) saved.push_back(detail::data_of(input));
    std::unique_ptr<std::mt19937_64> random;
    if (options.preserve_rng_state) random = std::make_unique<std::mt19937_64>(default_generator().engine());
    const detail::AutocastState precision;
    auto output = [&] {
        NoGradGuard no_grad;
        return function(inputs);
    }();
    if (!is_floating(output.dtype())) return output.detach();

    return detail::record(detail::data_of(output), std::move(edges),
        [saved = std::move(saved), function = std::move(function), random = std::move(random), precision]
        (const auto& seed, auto& sink, const auto& input_edges) {
            GradModeGuard recording(true);
            detail::AutocastReplay autocast(precision);
            RandomReplay replay(random.get());
            std::vector<Tensor> arguments;
            std::vector<detail::Edge> requests;
            arguments.reserve(saved.size());
            requests.reserve(saved.size());
            for (std::size_t index = 0; index < saved.size(); ++index) {
                // 每个输入槽建立独立叶身份，重复参数的各项偏导在外层沿原边汇合。
                arguments.push_back(detail::make_tensor(saved[index], sink.needs(input_edges[index])));
                requests.push_back(detail::gradient_edge(arguments.back()));
            }
            auto recomputed = function(arguments);
            const auto root = detail::gradient_edge(recomputed);
            auto contributions = detail::run_grad(std::span(&root, 1), requests, std::span(&seed, 1));
            for (std::size_t index = 0; index < contributions.size(); ++index) {
                if (contributions[index]) sink.add_owned(input_edges[index], std::move(*contributions[index]));
            }
        });
}

} // namespace dl
