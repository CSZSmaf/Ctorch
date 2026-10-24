#pragma once

#include "core/tensor_data.h"

#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace dl::detail {

struct Node;

struct Edge {
    std::shared_ptr<Node> node;
};

using NodeSet = std::unordered_set<const Node*>;

/// 叶梯度与 Tensor 身份分开拥有，累加节点只引用这个数据槽。
struct GradSlot {
    std::optional<TensorData> value;
    void accumulate(TensorData contribution);
};

/// 无初值供完整覆盖，零初值供局部写入与散射累计。
enum class GradInit { uninitialized, zero };

/// 已有目标继续累计，新目标按所选初值开始接收贡献。
struct GradTarget {
    TensorData& value;
    bool accumulate;
};

/// 每个累计目标拥有独立连续缓冲，首次贡献可由计算结果直接交接。
class GradSink {
public:
    explicit GradSink(const NodeSet* needed = nullptr) : needed_(needed) {}
    /// 普通 backward 接收全部有效边；梯度值调用只接收通向请求输入的边。
    bool needs(const Edge& edge) const {
        return edge.node && (!needed_ || needed_->contains(edge.node.get()));
    }
    /// 借用的贡献可与种子、保存值和其他输入共享；首次累计复制其逻辑值。
    void add(const Edge& edge, const TensorData& contribution);
    /// 交接独立、连续、可写的计算结果；调用方随后不再读写其存储。
    void add_owned(const Edge& edge, TensorData contribution);
    /// 借用广播后的贡献，按原输入形状直接归约到累计目标。
    void add_reduced(const Edge& edge, const TensorData& contribution, const Shape& shape);
    /// 为已选中的边取得连续目标；初值仅用于新目标，已有贡献始终保留。
    GradTarget target(const Edge& edge, const Shape& shape, DType dtype, Device device,
                      GradInit init = GradInit::uninitialized);
    /// 未收到贡献表示该分支的 VJP 为零，不为它分配梯度张量。
    std::optional<TensorData> take(const Node* node);

private:
    void accumulate(const Edge& edge, TensorData contribution, bool owns_storage);
    const NodeSet* needed_;
    std::unordered_map<const Node*, TensorData> gradients_;
};

struct Node {
    explicit Node(std::vector<Edge> inputs) : inputs(std::move(inputs)) {}
    virtual ~Node() = default;
    /// 引擎交接已汇合的梯度，普通 VJP 只读借用，叶节点直接接入梯度槽。
    virtual void backward(TensorData grad, GradSink& sink) = 0;
    virtual void release_saved() = 0;

    std::vector<Edge> inputs;
};

struct AutogradMeta {
    Edge edge;
    std::shared_ptr<GradSlot> leaf;
};

using SavedTensor = TensorData;

std::unique_ptr<AutogradMeta> make_leaf_meta();
void run_backward(const Edge& root, const TensorData& seed);
std::vector<std::optional<TensorData>> run_grad(std::span<const Edge> roots,
    std::span<const Edge> inputs, std::span<const TensorData> seeds);

} // namespace dl::detail
