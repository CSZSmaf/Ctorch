#include "dl/plan.h"

#include "core/amp.h"
#include "core/buffer_pool.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace dl::plan {

struct ExecutionPlan::Impl {
    struct Step {
        std::variant<Execute, MultiExecute> execute;
        std::vector<std::size_t> inputs, outputs;
        std::vector<std::size_t> release;
    };
    struct Binding { std::size_t input, slot; };
    struct Scratch {
        Scratch(std::size_t slots, std::size_t max_arguments) : values(slots) {
            arguments.reserve(max_arguments);
        }
        void clear() {
            arguments.clear();
            for (auto& value : values) value.reset();
        }
        std::vector<std::optional<Tensor>> values;
        std::vector<Tensor> arguments;
    };
    std::vector<TensorSpec> inputs, outputs;
    std::vector<Binding> bindings;
    std::vector<Step> steps;
    std::vector<std::size_t> output_slots;
    std::vector<StorageLifetime> storage;
    std::vector<BufferRequirement> buffers;
    std::size_t slots = 0, max_arguments = 0, fusions = 0;
    detail::AutocastState precision;

    std::vector<Tensor> run(std::span<const Tensor> inputs, Scratch& scratch) const;
};

struct ExecutionWorkspace::Impl {
    explicit Impl(std::shared_ptr<const ExecutionPlan::Impl> plan)
        : plan(std::move(plan)), scratch(this->plan->slots, this->plan->max_arguments) {
        for (const auto& requirement : this->plan->buffers) {
            pool.reserve(requirement.bytes, requirement.device, requirement.count);
        }
    }
    std::shared_ptr<const ExecutionPlan::Impl> plan;
    detail::BufferPool pool;
    ExecutionPlan::Impl::Scratch scratch;
};

Value Graph::input(TensorSpec spec, std::string name) {
    const Value result{values_.size()};
    values_.push_back({std::move(spec), std::nullopt});
    inputs_.push_back({result, std::move(name)});
    return result;
}

std::vector<TensorSpec> Graph::input_specs(std::span<const Value> inputs) const {
    std::vector<TensorSpec> specs;
    specs.reserve(inputs.size());
    for (const auto input : inputs) specs.push_back(spec(input));
    return specs;
}

std::vector<Value> Graph::append(std::string name, Invocation execute, Effect effect, Semantic semantic,
    std::span<const Value> inputs, std::vector<OutputSpec> outputs, OperatorInfo description) {
    std::vector<Value> results;
    results.reserve(outputs.size());
    for (auto& output : outputs) {
        std::optional<Value> alias;
        if (output.alias_input) alias = inputs[*output.alias_input];
        else if (output.alias_output) alias = results[*output.alias_output];
        results.push_back({values_.size()});
        values_.push_back({std::move(output.tensor), alias});
    }
    nodes_.push_back({std::move(name), std::move(execute), effect, semantic,
        {inputs.begin(), inputs.end()}, results, std::move(description)});
    return results;
}

Value Graph::call(Operation operation, std::span<const Value> inputs) {
    auto output = operation.infer(input_specs(inputs));
    return append(std::move(operation.name), std::move(operation.execute), operation.effect, operation.semantic,
        inputs, {std::move(output)}, std::move(operation.description))[0];
}

std::vector<Value> Graph::call(MultiOperation operation, std::span<const Value> inputs) {
    auto outputs = operation.infer(input_specs(inputs));
    return append(std::move(operation.name), std::move(operation.execute), operation.effect, Semantic::custom,
        inputs, std::move(outputs));
}

GraphInfo Graph::describe() const {
    GraphInfo info;
    for (const auto& value : values_) info.values.push_back(value.spec);
    for (const auto& input : inputs_) info.inputs.emplace_back(input.value, input.name);
    for (const auto& node : nodes_) info.nodes.push_back({node.name, node.description, node.effect, node.inputs, node.outputs});
    return info;
}

ExecutionPlan Graph::compile(std::span<const Value> outputs, CompileOptions options) const {
    auto plan = std::make_shared<ExecutionPlan::Impl>();

    // 任一结果被使用就保留整次调用；状态操作作为额外根参与同一次逆拓扑遍历。
    auto nodes = nodes_;
    std::vector<bool> needed(values_.size()), keep(nodes.size());
    for (const auto output : outputs) needed[output.index] = true;
    for (std::size_t i = nodes.size(); i-- > 0;) {
        const auto& node = nodes[i];
        keep[i] = node.effect != Effect::none || std::any_of(node.outputs.begin(), node.outputs.end(),
            [&](Value value) { return needed[value.index]; });
        if (keep[i]) for (const auto input : node.inputs) needed[input.index] = true;
    }

    if (options.fuse_pointwise) {
        // 输出引用也计作一次使用，因此公开返回的加法结果继续拥有自己的存储与求导节点。
        std::vector<std::size_t> uses(values_.size());
        for (const auto output : outputs) ++uses[output.index];
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (keep[i]) for (const auto input : nodes[i].inputs) ++uses[input.index];
        }
        std::optional<std::size_t> previous;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (!keep[i]) continue;
            auto& node = nodes[i];
            if (previous && node.semantic == Semantic::relu && node.effect == Effect::none) {
                const auto& producer = nodes[*previous];
                if (producer.semantic == Semantic::add && producer.effect == Effect::none
                    && node.inputs[0] == producer.outputs[0] && uses[node.inputs[0].index] == 1) {
                    // 只合并保留序列中的邻接操作，参数读取保持在相同状态操作之间。
                    node.name = "add_relu";
                    node.execute = ops::add_relu().execute;
                    node.inputs = producer.inputs;
                    node.semantic = Semantic::custom;
                    keep[*previous] = false;
                    ++plan->fusions;
                }
            }
            previous = i;
        }
    }

    // 改写后重新收集实际存在的值；被消除的中间结果不再占用槽或存储组。
    std::fill(needed.begin(), needed.end(), false);
    for (const auto output : outputs) needed[output.index] = true;
    std::vector<std::size_t> order, first(values_.size()), last(values_.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (!keep[i]) continue;
        const auto step = order.size();
        order.push_back(i);
        // 回调仍产生其全部结果；未被消费的结果在这一步结束后释放。
        for (const auto output : node.outputs) {
            needed[output.index] = true;
            first[output.index] = last[output.index] = step;
        }
        for (const auto input : node.inputs) {
            needed[input.index] = true;
            last[input.index] = step;
        }
    }
    for (const auto output : outputs) {
        last[output.index] = order.size();
        plan->outputs.push_back(spec(output));
    }

    // 一个值只登记一次释放位置，重复输入边仍分别传给实际算子和求导引擎。
    std::vector<std::vector<std::size_t>> retire(order.size());
    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (needed[i] && last[i] < order.size()) retire[last[i]].push_back(i);
    }
    std::vector<std::size_t> slots(values_.size()), available;
    auto allocate_slot = [&] {
        if (available.empty()) return plan->slots++;
        const auto slot = available.back();
        available.pop_back();
        return slot;
    };
    std::vector<bool> external(values_.size());
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
        const auto value = inputs_[i].value;
        plan->inputs.push_back(spec(value));
        external[value.index] = true;
        if (needed[value.index]) {
            slots[value.index] = allocate_slot();
            plan->bindings.push_back({i, slots[value.index]});
        }
    }
    for (std::size_t step = 0; step < order.size(); ++step) {
        const auto index = order[step];
        const auto& node = nodes[index];
        ExecutionPlan::Impl::Step invocation{node.execute, {}, {}, {}};
        for (const auto input : node.inputs) invocation.inputs.push_back(slots[input.index]);
        for (const auto output : node.outputs) {
            slots[output.index] = allocate_slot();
            invocation.outputs.push_back(slots[output.index]);
        }
        plan->max_arguments = std::max(plan->max_arguments, invocation.inputs.size());
        for (const auto value : retire[step]) {
            invocation.release.push_back(slots[value]);
            available.push_back(slots[value]);
        }
        plan->steps.push_back(std::move(invocation));
    }
    for (const auto output : outputs) plan->output_slots.push_back(slots[output.index]);

    // 别名链归并到原始存储值，使用区间取全部视图的并集。
    // 句柄槽可提前释放，Storage 共享所有权仍让存活视图和反向保存值持有数据。
    std::vector<std::size_t> storage_group(values_.size());
    for (std::size_t i = 0; i < values_.size(); ++i) {
        if (!needed[i]) continue;
        if (values_[i].alias) {
            storage_group[i] = storage_group[values_[i].alias->index];
            auto& lifetime = plan->storage[storage_group[i]];
            lifetime.last_step = std::max(lifetime.last_step, last[i]);
        } else {
            storage_group[i] = plan->storage.size();
            plan->storage.push_back({Value{i}, first[i], last[i], external[i]});
        }
    }
    // 同一步内先产生输出再释放输入，区间结束后的下一步才归还其尺寸类名额。
    // 元数据描述前向可见的连续存储，实际复用继续由 Storage 的共享引用决定。
    using BufferKey = std::tuple<DeviceType, int, std::size_t>;
    std::map<BufferKey, std::map<std::size_t, Index>> changes;
    for (const auto& lifetime : plan->storage) {
        const auto& tensor = spec(lifetime.origin);
        if (lifetime.external || !tensor.is_contiguous() || tensor.storage_offset != 0 || tensor.read_only) continue;
        const auto bytes = static_cast<std::size_t>(tensor.numel()) * element_size(tensor.dtype);
        auto& events = changes[{tensor.device.type, tensor.device.index, bytes}];
        ++events[lifetime.first_step];
        --events[lifetime.last_step + 1];
    }
    for (const auto& [key, events] : changes) {
        Index live = 0, peak = 0;
        for (const auto& [step, change] : events) {
            live += change;
            peak = std::max(peak, live);
        }
        plan->buffers.push_back({{std::get<0>(key), std::get<1>(key)}, std::get<2>(key),
            static_cast<std::size_t>(peak)});
    }
    return ExecutionPlan(std::move(plan));
}

std::span<const TensorSpec> ExecutionPlan::input_specs() const { return impl_->inputs; }
std::span<const TensorSpec> ExecutionPlan::output_specs() const { return impl_->outputs; }
std::span<const StorageLifetime> ExecutionPlan::storage_lifetimes() const { return impl_->storage; }
std::span<const BufferRequirement> ExecutionPlan::buffer_requirements() const { return impl_->buffers; }
std::size_t ExecutionPlan::step_count() const { return impl_->steps.size(); }
std::size_t ExecutionPlan::slot_count() const { return impl_->slots; }
std::size_t ExecutionPlan::fusion_count() const { return impl_->fusions; }

std::vector<Tensor> ExecutionPlan::Impl::run(std::span<const Tensor> inputs, Scratch& scratch) const {
    detail::AutocastReplay autocast(precision);
    // 参数和槽只服务本次调用；正常返回及异常展开都在此处结束临时句柄的持有。
    struct Frame {
        Scratch& scratch;
        ~Frame() { scratch.clear(); }
    } frame{scratch};
    auto& values = scratch.values;
    auto& arguments = scratch.arguments;
    for (const auto& binding : bindings) values[binding.slot] = inputs[binding.input];
    for (const auto& step : steps) {
        for (const auto slot : step.inputs) arguments.push_back(*values[slot]);
        if (const auto* execute = std::get_if<Execute>(&step.execute)) {
            values[step.outputs[0]] = (*execute)(arguments);
        } else {
            auto results = std::get<MultiExecute>(step.execute)(arguments);
            for (std::size_t i = 0; i < step.outputs.size(); ++i) {
                values[step.outputs[i]] = std::move(results[i]);
            }
        }
        arguments.clear();
        for (const auto slot : step.release) values[slot].reset();
    }
    std::vector<Tensor> outputs;
    outputs.reserve(output_slots.size());
    for (const auto slot : output_slots) outputs.push_back(*values[slot]);
    return outputs;
}

std::vector<Tensor> ExecutionPlan::operator()(std::span<const Tensor> inputs) const {
    Impl::Scratch scratch(impl_->slots, impl_->max_arguments);
    return impl_->run(inputs, scratch);
}

Tensor ExecutionPlan::checkpoint(std::span<const Tensor> inputs, CheckpointOptions options) const {
    return dl::checkpoint(inputs, [plan = *this](std::span<const Tensor> values) {
        auto outputs = plan(values);
        return std::move(outputs.front());
    }, options);
}

ExecutionWorkspace ExecutionPlan::prepare() const {
    return ExecutionWorkspace(std::make_unique<ExecutionWorkspace::Impl>(impl_));
}

ExecutionWorkspace::ExecutionWorkspace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExecutionWorkspace::~ExecutionWorkspace() = default;
ExecutionWorkspace::ExecutionWorkspace(ExecutionWorkspace&&) noexcept = default;
ExecutionWorkspace& ExecutionWorkspace::operator=(ExecutionWorkspace&&) noexcept = default;

std::vector<Tensor> ExecutionWorkspace::operator()(std::span<const Tensor> inputs) {
    return with_memory([&] { return impl_->plan->run(inputs, impl_->scratch); });
}

void* ExecutionWorkspace::enter_memory() noexcept { return detail::exchange_buffer_pool(&impl_->pool); }
void ExecutionWorkspace::leave_memory(void* previous) noexcept {
    detail::exchange_buffer_pool(static_cast<detail::BufferPool*>(previous));
}

WorkspaceMemoryStats ExecutionWorkspace::memory_stats(Device device) const {
    const auto stats = impl_->pool.stats(device);
    return {stats.buffers, stats.in_use_buffers, stats.reserved_bytes, stats.in_use_bytes};
}

void ExecutionWorkspace::release_unused_memory() { impl_->pool.release_unused(); }

} // namespace dl::plan
