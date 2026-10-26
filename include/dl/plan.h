#pragma once

#include "dl/autograd.h"
#include "dl/extension.h"
#include "dl/options.h"
#include "dl/optim.h"

#include <functional>
#include <string>
#include <utility>
#include <variant>

namespace dl::nn { class FunctionalModule; }

namespace dl::plan {

/// Value 只在创建它的 Graph 中有意义；节点引用此前创建的值，插入顺序即拓扑顺序。
struct Value {
    std::size_t index;
    bool operator==(const Value&) const = default;
};

/// 输出可以共享某个输入，或同一次调用中位于它之前的输出；两个别名字段至多设置一个。
/// 索引对应本次操作的输入/输出列表，无别名时拥有独立存储。
struct OutputSpec {
    TensorSpec tensor;
    std::optional<std::size_t> alias_input = std::nullopt;
    std::optional<std::size_t> alias_output = std::nullopt;
};

enum class Effect { none, random, mutation, external };
/// 语义标记供编译期改写使用，与显示名称无关；custom 保留回调的原有调用方式。
/// add/relu 分别承诺公开 dl::add/dl::relu 的规格、别名和一阶求导语义。
enum class Semantic { custom, add, relu };
using Attribute = std::pair<std::string, std::variant<Index, double, Shape>>;
/// 可序列化的算子语义与常量属性。kind 独立于显示名称，空值表示只有 C++ 回调。
struct OperatorInfo { std::string kind; std::vector<Attribute> attributes; };
struct NodeInfo {
    std::string name;
    OperatorInfo operation;
    Effect effect;
    std::vector<Value> inputs, outputs;
};
struct GraphInfo {
    std::vector<TensorSpec> values;
    std::vector<std::pair<Value, std::string>> inputs;
    std::vector<NodeInfo> nodes;
};
using Infer = std::function<OutputSpec(std::span<const TensorSpec>)>;
using Execute = std::function<Tensor(std::span<const Tensor>)>;

/// 普通 callable 组成一个单输出操作，execute 复用 eager 算子或 custom_vjp。
/// infer 只处理规格，不读取数据或改变状态；execute 的输出规格与推导结果相同。
/// none 表示只读显式输入且没有可观察状态影响；其余操作即使输出未使用也保留。
struct Operation {
    std::string name;
    Infer infer;
    Execute execute;
    Effect effect = Effect::none;
    Semantic semantic = Semantic::custom;
    OperatorInfo description = {};
};

using MultiInfer = std::function<std::vector<OutputSpec>(std::span<const TensorSpec>)>;
using MultiExecute = std::function<std::vector<Tensor>(std::span<const Tensor>)>;

/// 一次调用交付固定数量的结果，结果数量和顺序与 infer 一致，各结果保留自己的 eager 图。
/// 零输出可表达只有状态影响的操作；单输出 Operation 继续直接返回 Tensor。
struct MultiOperation {
    std::string name;
    MultiInfer infer;
    MultiExecute execute;
    Effect effect = Effect::none;
};

/// 前向可见存储组的使用区间；视图归入原始值，输入存储标记为 external。
/// 下标为保留后的操作序号，last_step == step_count() 表示输出继续持有存储。
/// 求导保存值、内核工作区和调用方持有的其他别名拥有各自的生命周期。
struct StorageLifetime {
    Value origin;
    std::size_t first_step, last_step;
    bool external;
};

struct CompileOptions {
    /// 合并单一消费者的相邻 add -> ReLU，减少一次调用和整张量中间值。
    bool fuse_pointwise = true;
};

/// 独立连续输出的前向区间按设备和精确字节数合并，count 是该尺寸的同时存活峰值。
/// 反向保存值、内核临时分配和跨次持有的结果可能使运行时需要更多块。
struct BufferRequirement {
    Device device;
    std::size_t bytes, count;
};

struct WorkspaceMemoryStats {
    std::size_t buffers = 0, in_use_buffers = 0;
    std::size_t reserved_bytes = 0, in_use_bytes = 0;
};

class Graph;
class ExecutionWorkspace;

/// 固定输入规格的可复制执行计划，保存调用结构而不保存参数数值或历史求导图。
/// 每次调用重新绑定全部输入，参数仍使用原 Tensor 身份；梯度模式取自调用现场。
class DL_API ExecutionPlan {
public:
    std::span<const TensorSpec> input_specs() const;
    std::span<const TensorSpec> output_specs() const;
    std::span<const StorageLifetime> storage_lifetimes() const;
    std::span<const BufferRequirement> buffer_requirements() const;
    std::size_t step_count() const;
    std::size_t slot_count() const;
    std::size_t fusion_count() const;

    /// 预留前向尺寸类的存储与宿主槽，返回可串行复用的独立工作区。
    /// 此调用实际分配 CPU/CUDA 内存，数值计算仍在工作区调用时发生。
    ExecutionWorkspace prepare() const;

    /// 输入顺序、shape/stride/dtype/device/offset/read_only 与 input_specs 一致。
    /// 返回顺序对应编译时的输出列表；重复输出保留同一 Tensor 身份。
    std::vector<Tensor> operator()(std::span<const Tensor> inputs) const;
    std::vector<Tensor> operator()(std::initializer_list<Tensor> inputs) const {
        return (*this)(std::span<const Tensor>(inputs.begin(), inputs.size()));
    }

    /// 将单个浮点输出的计划作为激活重算块，所有参数继续从输入槽读取。
    /// 计划只包含只读计算及可重放的随机采样，持久状态更新放在块外。
    /// 节点持有不可变计划，初次前向与反向重算均使用编译时选定的精度和融合。
    Tensor checkpoint(std::span<const Tensor> inputs, CheckpointOptions options = {}) const;
    Tensor checkpoint(std::initializer_list<Tensor> inputs, CheckpointOptions options = {}) const {
        return checkpoint(std::span<const Tensor>(inputs.begin(), inputs.size()), options);
    }

private:
    struct Impl;
    explicit ExecutionPlan(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
    friend class Graph;
    friend class ExecutionWorkspace;
};

/// 工作区共享不可变计划，独立持有缓冲与宿主槽；每次调用重新绑定输入。
/// 结果和求导保存值拥有普通 Storage 引用，可跨调用或超出工作区生命周期持有。
/// 同一工作区的使用串行安排，前向执行不递归进入自身；分配作用域可嵌套。
/// 并发执行使用独立工作区，工作区在其分配作用域内保持存活。
class DL_API ExecutionWorkspace {
public:
    ~ExecutionWorkspace();
    ExecutionWorkspace(ExecutionWorkspace&&) noexcept;
    ExecutionWorkspace& operator=(ExecutionWorkspace&&) noexcept;
    ExecutionWorkspace(const ExecutionWorkspace&) = delete;
    ExecutionWorkspace& operator=(const ExecutionWorkspace&) = delete;

    std::vector<Tensor> operator()(std::span<const Tensor> inputs);
    std::vector<Tensor> operator()(std::initializer_list<Tensor> inputs) {
        return (*this)(std::span<const Tensor>(inputs.begin(), inputs.size()));
    }
    /// 在当前线程同步调用 function，让其中的普通张量分配复用此工作区。
    /// 可包含前向、损失、反向和更新；返回值保持原类型及普通张量所有权。
    /// 作用域只选择缓冲池，梯度与精度模式仍由各自接口控制。
    template <class Function> decltype(auto) with_memory(Function&& function) {
        MemoryScope scope(*this);
        return std::invoke(std::forward<Function>(function));
    }
    /// 统计该设备在此工作区中的 Storage 容量；in_use 表示工作区外仍有引用。
    WorkspaceMemoryStats memory_stats(Device device = Device::cpu()) const;
    /// 释放各设备没有其他持有者的块；CUDA 沿计算 stream 归还，不等待设备。
    void release_unused_memory();

private:
    struct Impl;
    void* enter_memory() noexcept;
    static void leave_memory(void* previous) noexcept;
    // 栈上的作用域只保存外层池，正常返回和异常展开共用同一恢复过程。
    class MemoryScope {
    public:
        explicit MemoryScope(ExecutionWorkspace& workspace) noexcept : previous_(workspace.enter_memory()) {}
        ~MemoryScope() { ExecutionWorkspace::leave_memory(previous_); }
        MemoryScope(const MemoryScope&) = delete;
        MemoryScope& operator=(const MemoryScope&) = delete;
    private:
        void* previous_;
    };
    explicit ExecutionWorkspace(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class ExecutionPlan;
};

/// 显式构图只推导宿主元数据；数值计算在 ExecutionPlan 调用时发生。
/// 参数、buffer 和输入都可放入输入槽，规格或执行路径变化时构建相应的新图。
class DL_API Graph {
public:
    Value input(TensorSpec spec, std::string name = {});
    Value call(Operation operation, std::span<const Value> inputs);
    Value call(Operation operation, std::initializer_list<Value> inputs) {
        return call(std::move(operation), std::span<const Value>(inputs.begin(), inputs.size()));
    }
    std::vector<Value> call(MultiOperation operation, std::span<const Value> inputs);
    std::vector<Value> call(MultiOperation operation, std::initializer_list<Value> inputs) {
        return call(std::move(operation), std::span<const Value>(inputs.begin(), inputs.size()));
    }
    const TensorSpec& spec(Value value) const { return values_[value.index].spec; }
    /// 复制图的规格和算子描述，供导出器与可视化工具读取，不持有执行回调。
    GraphInfo describe() const;

    /// 删除未使用的纯计算，保留全部状态操作及其依赖，随后融合并规划生命周期。
    /// 记录当前线程的 CPU/CUDA autocast 配置，每次执行在此精度作用域内运行。
    ExecutionPlan compile(std::span<const Value> outputs, CompileOptions options = {}) const;
    ExecutionPlan compile(std::initializer_list<Value> outputs, CompileOptions options = {}) const {
        return compile(std::span<const Value>(outputs.begin(), outputs.size()), options);
    }

private:
    using Invocation = std::variant<Execute, MultiExecute>;
    struct ValueInfo { TensorSpec spec; std::optional<Value> alias; };
    struct Input { Value value; std::string name; };
    struct Node {
        std::string name;
        Invocation execute;
        Effect effect;
        Semantic semantic;
        std::vector<Value> inputs, outputs;
        OperatorInfo description;
    };
    std::vector<TensorSpec> input_specs(std::span<const Value> inputs) const;
    std::vector<Value> append(std::string name, Invocation execute, Effect effect, Semantic semantic,
        std::span<const Value> inputs, std::vector<OutputSpec> outputs, OperatorInfo description = {});
    std::vector<ValueInfo> values_;
    std::vector<Input> inputs_;
    std::vector<Node> nodes_;
};

/// 内置描述复用布局函数和 eager 计算；自定义算子可直接构造 Operation。
namespace ops {
DL_API Operation add();
DL_API Operation add_relu();
DL_API Operation mul();
DL_API Operation sub();
DL_API Operation div();
DL_API Operation sigmoid();
DL_API Operation tanh();
DL_API Operation softmax(Index axis = -1);
/// 输入为 x、可选 weight、可选 bias；仿射参数依次给出。
DL_API Operation layer_norm(Shape normalized_shape, double epsilon = 1e-5);
DL_API Operation matmul();
/// 输入依次为 [..., in]、[out, in] 和可选的 [out] 偏置，结果保留前导维。
DL_API Operation linear();
/// 一维输入为 [N,C,L]/[C,L]，二维为 NCHW；之后依次传入分组权重和可选通道偏置。
/// 卷积结果整理为规范连续布局，使计划规格独立于后端和 autocast 的物理布局选择。
DL_API Operation conv1d(Conv1dOptions options = {});
DL_API Operation conv2d(Conv2dOptions options = {});
/// 固定窗口池化接收一个输入，形状、填充计数与求导规则沿用对应 eager 算子。
DL_API Operation max_pool1d(Pool1dOptions options = {});
DL_API Operation avg_pool1d(Pool1dOptions options = {}, bool count_include_pad = true);
DL_API Operation max_pool2d(Pool2dOptions options = {});
DL_API Operation avg_pool2d(Pool2dOptions options = {}, bool count_include_pad = true);
/// 输出空间尺寸固定在描述中，保留输入的通道维和可选批维。
DL_API Operation adaptive_avg_pool1d(Index output_size);
DL_API Operation adaptive_avg_pool2d(Size2 output_size);
/// 接收 NCHW 或 CHW 图像，输出空间尺寸、插值模式和角点对齐规则固定在描述中。
DL_API Operation interpolate(Size2 output_size, InterpolationMode mode = InterpolationMode::nearest,
                              bool align_corners = false);
DL_API Operation relu();
DL_API Operation silu();
DL_API Operation reshape(Shape shape);
DL_API Operation transpose(Index first, Index second);
DL_API Operation contiguous();
/// 使用默认 Generator；训练模式和概率固定在描述中，0 < p < 1 时保留随机影响。
DL_API Operation dropout(double probability, bool training = true);
/// 输入顺序为 input、运行 mean/variance（启用跟踪时）、weight/bias（启用 affine 时）。
/// 训练且跟踪统计时直接更新输入状态，声明 mutation；其他模式为只读计算。
DL_API Operation batch_norm(BatchNormOptions options = {}, bool training = true);
/// 输入顺序同 batch_norm，返回 output、next_mean、next_variance；不跟踪统计时只返回 output。
/// 训练创建独立新状态，eval 返回输入状态别名；全部结果通过显式数据边传递。
DL_API MultiOperation functional_batch_norm(BatchNormOptions options = {}, bool training = true);
/// 输入为前向实参、模块 parameters、buffers 和最后一个 CPU int64[2] key；模式固定在描述中。
/// infer 读取完整输入规格，其中 buffer 规格已调整为本次副本的连续布局；只描述模型输出。
/// forward 在绑定作用域内接收前向实参；原模块及回调引用的对象在计划使用期间保持存活。
/// 结果为 next_buffers 后接模型输出；infer 的 buffer 输入别名映射到对应的 next_buffer，
/// 同级模型输出别名按其原顺序填写。回调只读外部状态，随机计算使用显式绑定 key。
DL_API MultiOperation functional_module(nn::FunctionalModule module, std::vector<std::uint8_t> training,
                                       MultiInfer infer, MultiExecute forward);
/// 沿指定轴逐项返回视图，输出数量由输入规格确定，所有结果共享输入存储。
DL_API MultiOperation unbind(Index axis = 0);
/// 输入为 input、hidden、cell、weight_ih、weight_hh，以及 bias=true 时的 bias_ih/bias_hh。
/// 输出依次为 next_hidden、next_cell，求导规则复用 eager LSTM 单步。
DL_API MultiOperation lstm_cell(bool bias = true);
/// 随机 key 使用 CPU 连续 int64[2] 输入；这些操作只读 key，均为纯计算。
DL_API Operation fold_in_key(std::uint64_t label);
DL_API MultiOperation split_key(std::size_t count = 2);
/// 工厂接收一个 key 输入，数据输出按 options/device 放在 CPU 或 CUDA。
DL_API Operation uniform(Shape shape, double low, double high, TensorOptions options = {});
DL_API Operation randn(Shape shape, TensorOptions options = {});
DL_API Operation bernoulli(Shape shape, double probability, Device device = Device::cpu());
/// 输入依次为数据和 CPU key，输出与有状态 Dropout 使用相同数学和求导规则。
DL_API Operation dropout_with_key(double probability, bool training = true);
/// 输入依次为 Q/K/V，masked 时追加 bool mask，keyed 时最后追加 CPU int64[2] key。
/// 数学或 cuDNN 路径由 options.backend 显式选择，规格遵循相应公开注意力入口。
DL_API Operation attention(AttentionComputeOptions options = {}, bool masked = false, bool keyed = false);
/// 函数式更新为纯计算，新参数和状态通过多输出传递；梯度是显式输入。
/// SGD 输入为 parameter、gradient，以及可选 velocity；输出为 next_parameter 和可用的 next_velocity。
DL_API MultiOperation sgd(optim::SGDOptions options = {});
/// AdamW 输入为 parameter、gradient，已有状态时追加 first、second、CPU int64 标量 step。
/// 输出为 next_parameter、next_first、next_second、CPU int64 标量 next_step。
DL_API MultiOperation adamw(optim::AdamWOptions options = {});
/// Adam 使用与 AdamW 相同的输入/输出顺序，权重衰减进入矩估计。
DL_API MultiOperation adam(optim::AdamOptions options = {});
/// RMSprop 的已有状态输入追加 square_average、启用的 momentum/gradient_average、CPU int64 step。
/// 输出为 next_parameter、next_square_average、相同次序的启用缓冲、next_step。
/// 描述的可选槽由固定 options 选择，停用缓冲由调用方单独保留。
DL_API MultiOperation rmsprop(optim::RMSpropOptions options = {});
/// Adagrad 的已有状态输入追加 sum、CPU int64 step；输出为 next_parameter、next_sum、next_step。
DL_API MultiOperation adagrad(optim::AdagradOptions options = {});
} // namespace ops

} // namespace dl::plan
