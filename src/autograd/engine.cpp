#include "autograd/graph.h"

#include "dl/autograd.h"
#include "ops/numeric.h"

#include <algorithm>

namespace dl {
namespace {
thread_local bool grad_enabled = true;
}

bool is_grad_enabled() { return grad_enabled; }
GradModeGuard::GradModeGuard(bool enabled) : previous_(grad_enabled) { grad_enabled = enabled; }
GradModeGuard::~GradModeGuard() { grad_enabled = previous_; }
NoGradGuard::NoGradGuard() : mode_(false) {}
NoGradGuard::~NoGradGuard() = default;

namespace detail {
namespace {

class AccumulateGrad final : public Node {
public:
    explicit AccumulateGrad(std::shared_ptr<GradSlot> slot)
        : Node(std::vector<Edge>{}), slot_(std::move(slot)) {}
    void backward(TensorData grad, GradSink&) override { slot_->accumulate(std::move(grad)); }
    void release_saved() override {}

private:
    std::shared_ptr<GradSlot> slot_;
};

struct NodeTask {
    std::shared_ptr<Node> node;
    bool propagate = true;
};

using CapturedGradients = std::unordered_map<const Node*, std::optional<TensorData>>;

std::vector<NodeTask> topological_order(std::span<const Edge> roots) {
    std::vector<NodeTask> order;
    NodeSet visited;
    std::vector<std::pair<std::shared_ptr<Node>, bool>> pending;
    for (const auto& root : roots) {
        if (root.node) pending.emplace_back(root.node, false);
    }
    while (!pending.empty()) {
        auto [node, expanded] = std::move(pending.back());
        pending.pop_back();
        if (expanded) {
            order.push_back({std::move(node)});
        } else if (visited.insert(node.get()).second) {
            pending.emplace_back(node, true);
            for (const auto& input : node->inputs) {
                if (input.node) {
                    pending.emplace_back(input.node, false);
                }
            }
        }
    }
    return order;
}

void execute_backward(std::vector<NodeTask>& order, GradSink& sink,
                      CapturedGradients* captured = nullptr) {
    // 逆拓扑顺序汇合所有根和重复边的贡献，中间结果与叶都在同一位置交付完整梯度。
    for (auto position = order.rbegin(); position != order.rend(); ++position) {
        auto& node = *position->node;
        if (auto grad = sink.take(&node)) {
            if (captured) {
                if (auto entry = captured->find(&node); entry != captured->end())
                    entry->second = *grad;
            }
            if (position->propagate) node.backward(std::move(*grad), sink);
        }
        // 数值反向可在请求的输入处结束，图保存值仍在本次消费中统一释放。
        node.release_saved();
        node.inputs.clear();
    }
}

} // namespace

void GradSlot::accumulate(TensorData contribution) {
    if (value) {
        numeric::add_into(*value, contribution);
    } else {
        value = std::move(contribution);
    }
}

void GradSink::add(const Edge& edge, const TensorData& contribution) {
    accumulate(edge, contribution, false);
}

void GradSink::add_owned(const Edge& edge, TensorData contribution) {
    accumulate(edge, std::move(contribution), true);
}

void GradSink::add_reduced(const Edge& edge, const TensorData& contribution, const Shape& shape) {
    if (!needs(edge)) return;
    auto output = target(edge, shape, contribution.dtype, contribution.device());
    numeric::sum_to_shape_into(output.value, contribution, output.accumulate);
}

GradTarget GradSink::target(const Edge& edge, const Shape& shape, DType dtype, Device device, GradInit init) {
    auto position = gradients_.find(edge.node.get());
    if (position != gradients_.end()) return {position->second, true};
    dtype = accumulation_dtype(dtype);
    auto value = init == GradInit::zero ? numeric::full(shape, 0.0, dtype, device)
                                        : empty_data(shape, dtype, device);
    auto entry = gradients_.emplace(edge.node.get(), std::move(value)).first;
    return {entry->second, false};
}

void GradSink::accumulate(const Edge& edge, TensorData contribution, bool owns_storage) {
    if (!needs(edge)) return;
    const auto dtype = accumulation_dtype(contribution.dtype);
    if (contribution.dtype != dtype) {
        // 类型转换已生成独立连续结果，可直接成为该目标的累计缓冲。
        contribution = numeric::cast(contribution, dtype);
        owns_storage = true;
    }
    // try_emplace 仅为新目标移动贡献；已有目标继续读取本地贡献并原地相加。
    auto [position, inserted] = gradients_.try_emplace(edge.node.get(), std::move(contribution));
    if (!inserted) numeric::add_into(position->second, contribution);
    else if (!owns_storage) position->second = numeric::clone(position->second);
}

std::optional<TensorData> GradSink::take(const Node* node) {
    auto entry = gradients_.extract(node);
    if (entry.empty()) return std::nullopt;
    return std::move(entry.mapped());
}

std::unique_ptr<AutogradMeta> make_leaf_meta() {
    auto slot = std::make_shared<GradSlot>();
    auto node = std::make_shared<AccumulateGrad>(slot);
    return std::make_unique<AutogradMeta>(AutogradMeta{{std::move(node)}, std::move(slot)});
}

void run_backward(const Edge& root, const TensorData& seed) {
    auto order = topological_order(std::span<const Edge>(&root, 1));
    GradSink sink;
    sink.add(root, seed);
    execute_backward(order, sink);
}

std::vector<std::optional<TensorData>> run_grad(std::span<const Edge> roots,
    std::span<const Edge> inputs, std::span<const TensorData> seeds) {
    auto order = topological_order(roots);
    CapturedGradients captured;
    NodeSet needed;
    for (const auto& input : inputs) {
        if (input.node) {
            captured.try_emplace(input.node.get());
            needed.insert(input.node.get());
        }
    }

    // 正拓扑遍历标记从请求输入到输出的路径。叶只交付数值，永不进入累加节点。
    // 请求中间结果时，只有另一个请求位于其上游，才继续执行该结果的 VJP。
    for (auto& task : order) {
        task.propagate = std::any_of(task.node->inputs.begin(), task.node->inputs.end(),
            [&](const Edge& edge) { return needed.contains(edge.node.get()); });
        if (task.propagate) needed.insert(task.node.get());
    }
    GradSink sink(&needed);
    for (std::size_t i = 0; i < roots.size(); ++i) sink.add(roots[i], seeds[i]);
    execute_backward(order, sink, &captured);

    std::vector<std::optional<TensorData>> result;
    result.reserve(inputs.size());
    for (const auto& input : inputs)
        result.push_back(input.node ? captured.at(input.node.get()) : std::nullopt);
    return result;
}

} // namespace detail
} // namespace dl
