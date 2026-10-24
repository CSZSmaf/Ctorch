#include "dl/plan_c_api.h"

#include "api/callback.h"

#include <stdexcept>
#include <type_traits>

struct dl_graph { dl::plan::Graph value; };
struct dl_plan_operation { std::variant<dl::plan::Operation, dl::plan::MultiOperation> value; };
struct dl_plan_values { std::vector<dl::plan::Value> value; };
struct dl_execution_plan { dl::plan::ExecutionPlan value; };
struct dl_execution_workspace { dl::plan::ExecutionWorkspace value; };
struct dl_plan_infer_context { std::vector<dl::plan::OutputSpec> value; };

using namespace dl::capi;

namespace {

std::vector<dl::plan::Value> values(const dl_plan_value* inputs, std::size_t count) {
    std::vector<dl::plan::Value> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) result.push_back({inputs[i].index});
    return result;
}

std::optional<std::size_t> alias(int64_t index) {
    return index < 0 ? std::nullopt : std::optional{static_cast<std::size_t>(index)};
}

dl::plan::MultiInfer inference(dl_plan_infer_fn infer, std::shared_ptr<void> state) {
    return [infer, state = std::move(state)](std::span<const dl::TensorSpec> inputs) {
        std::vector<dl_tensor_spec> arguments;
        arguments.reserve(inputs.size());
        for (const auto& spec : inputs) arguments.push_back(convert(spec));
        dl_plan_infer_context context;
        const char* error = nullptr;
        if (infer(&context, arguments.data(), arguments.size(), state.get(), &error) != DL_OK)
            throw std::runtime_error(error ? error : "plan shape inference failed");
        return std::move(context.value);
    };
}

} // namespace

extern "C" {

dl_status dl_plan_builtin_operation(dl_plan_builtin kind, dl_plan_operation** output) {
    using Factory = dl::plan::Operation (*)();
    static constexpr Factory factories[]{dl::plan::ops::add, dl::plan::ops::add_relu,
        dl::plan::ops::mul, dl::plan::ops::matmul, dl::plan::ops::relu,
        dl::plan::ops::silu, dl::plan::ops::contiguous, dl::plan::ops::linear};
    return make_handle(output, [&] { return factories[static_cast<std::size_t>(kind)](); });
}

dl_status dl_plan_conv1d(dl_conv1d_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::conv1d(convert(options)); });
}
dl_status dl_plan_conv2d(dl_conv2d_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::conv2d(convert(options)); });
}
dl_status dl_plan_max_pool1d(dl_pool1d_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::max_pool1d(convert(options)); });
}
dl_status dl_plan_avg_pool1d(dl_pool1d_options options, int32_t count_include_pad, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::avg_pool1d(convert(options), count_include_pad != 0); });
}
dl_status dl_plan_max_pool2d(dl_pool2d_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::max_pool2d(convert(options)); });
}
dl_status dl_plan_avg_pool2d(dl_pool2d_options options, int32_t count_include_pad, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::avg_pool2d(convert(options), count_include_pad != 0); });
}
dl_status dl_plan_adaptive_avg_pool1d(int64_t output_size, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::adaptive_avg_pool1d(output_size); });
}
dl_status dl_plan_adaptive_avg_pool2d(dl_size2 output_size, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::adaptive_avg_pool2d(convert(output_size)); });
}
dl_status dl_plan_interpolate(dl_size2 output_size, dl_interpolation_mode mode, int32_t align_corners,
    dl_plan_operation** output) {
    return make_handle(output, [&] {
        return dl::plan::ops::interpolate(convert(output_size), convert(mode), align_corners != 0);
    });
}

dl_status dl_plan_reshape(dl_shape shape, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::reshape(convert(shape)); });
}
dl_status dl_plan_transpose(int64_t first, int64_t second, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::transpose(first, second); });
}
dl_status dl_plan_dropout(double probability, int32_t training, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::dropout(probability, training != 0); });
}
dl_status dl_plan_batch_norm(dl_batch_norm_options options, int32_t training, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::batch_norm(convert(options), training != 0); });
}
dl_status dl_plan_functional_batch_norm(dl_batch_norm_options options, int32_t training,
    dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::functional_batch_norm(convert(options), training != 0); });
}

dl_status dl_plan_functional_module(const dl_functional* functional, const uint8_t* training, size_t count,
    dl_plan_infer_fn infer, void* user_data, dl_user_data_deleter deleter, dl_plan_operation** output) {
    *output = nullptr;
    return invoke([&] {
        auto state = callback_state(user_data, deleter);
        auto operation = dl::plan::ops::functional_module(functional->value, {training, training + count},
            inference(infer, state),
            // 推导回调的上下文与模型一起由执行描述持有，随图和计划交接。
            [owner = functional->owner, state](std::span<const dl::Tensor> inputs) {
                return std::vector<dl::Tensor>{(*owner)(inputs[0])};
            });
        *output = new dl_plan_operation{std::move(operation)};
    });
}
dl_status dl_plan_unbind(int64_t axis, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::unbind(axis); });
}
dl_status dl_plan_lstm_cell(int32_t bias, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::lstm_cell(bias != 0); });
}
dl_status dl_plan_fold_in_key(uint64_t label, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::fold_in_key(label); });
}
dl_status dl_plan_split_key(size_t count, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::split_key(count); });
}
dl_status dl_plan_uniform(dl_shape shape, double low, double high, dl_tensor_options options,
    dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::uniform(convert(shape), low, high, convert(options)); });
}
dl_status dl_plan_randn(dl_shape shape, dl_tensor_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::randn(convert(shape), convert(options)); });
}
dl_status dl_plan_bernoulli(dl_shape shape, double probability, dl_device device, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::bernoulli(convert(shape), probability, convert(device)); });
}
dl_status dl_plan_dropout_with_key(double probability, int32_t training, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::dropout_with_key(probability, training != 0); });
}
dl_status dl_plan_attention(dl_attention_compute_options options, int32_t masked, int32_t keyed,
    dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::attention(convert(options), masked != 0, keyed != 0); });
}
dl_status dl_plan_sgd(dl_sgd_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::sgd(convert(options)); });
}
dl_status dl_plan_adamw(dl_adamw_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::adamw(convert(options)); });
}
dl_status dl_plan_adam(dl_adam_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::adam(convert(options)); });
}
dl_status dl_plan_rmsprop(dl_rmsprop_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::rmsprop(convert(options)); });
}
dl_status dl_plan_adagrad(dl_adagrad_options options, dl_plan_operation** output) {
    return make_handle(output, [&] { return dl::plan::ops::adagrad(convert(options)); });
}

dl_status dl_plan_custom_operation(const char* name, dl_plan_infer_fn infer,
    dl_plan_execute_fn execute, dl_plan_effect effect, void* user_data,
    dl_user_data_deleter deleter, dl_plan_operation** output) {
    *output = nullptr;
    return invoke([&] {
        auto state = callback_state(user_data, deleter);
        dl::plan::MultiOperation operation{
            name ? name : "",
            inference(infer, state),
            [execute, state](std::span<const dl::Tensor> inputs) {
                BorrowedTensors arguments(inputs);
                ReturnedHandle<dl_tensor_list> result;
                const char* error = nullptr;
                if (execute(arguments.data(), arguments.size(), state.get(), &result.value, &error) != DL_OK)
                    throw std::runtime_error(error ? error : "plan execution callback failed");
                return std::move(result.value->value);
            },
            static_cast<dl::plan::Effect>(effect)};
        *output = new dl_plan_operation{std::move(operation)};
    });
}

dl_status dl_plan_infer_add_output(dl_plan_infer_context* context, dl_plan_output_spec spec) {
    return invoke([&] {
        context->value.push_back({convert(spec.tensor), alias(spec.alias_input), alias(spec.alias_output)});
    });
}
dl_status dl_plan_infer_contiguous_output(dl_plan_infer_context* context,
    dl_shape shape, dl_dtype dtype, dl_device device) {
    return invoke([&] {
        context->value.push_back({dl::contiguous_spec(convert(shape), convert(dtype), convert(device))});
    });
}
void dl_plan_operation_release(dl_plan_operation* operation) { delete operation; }

dl_status dl_graph_create(dl_graph** output) {
    return make_handle(output, [] { return dl::plan::Graph{}; });
}
void dl_graph_release(dl_graph* graph) { delete graph; }
dl_status dl_graph_input(dl_graph* graph, dl_tensor_spec spec, const char* name, dl_plan_value* output) {
    return invoke([&] { *output = {graph->value.input(convert(spec), name ? name : "").index}; });
}
dl_status dl_graph_input_contiguous(dl_graph* graph, dl_shape shape, dl_dtype dtype,
    dl_device device, const char* name, dl_plan_value* output) {
    return invoke([&] {
        *output = {graph->value.input(dl::contiguous_spec(convert(shape), convert(dtype), convert(device)),
            name ? name : "").index};
    });
}
dl_tensor_spec dl_graph_get_spec(const dl_graph* graph, dl_plan_value value) {
    return convert(graph->value.spec({value.index}));
}
dl_status dl_graph_call(dl_graph* graph, const dl_plan_operation* operation,
    const dl_plan_value* inputs, size_t count, dl_plan_values** output) {
    return make_handle(output, [&] {
        auto arguments = values(inputs, count);
        return std::visit([&](const auto& description) {
            if constexpr (std::is_same_v<std::decay_t<decltype(description)>, dl::plan::Operation>) {
                return std::vector{graph->value.call(description, arguments)};
            } else {
                return graph->value.call(description, arguments);
            }
        }, operation->value);
    });
}
void dl_plan_values_release(dl_plan_values* values) { delete values; }
size_t dl_plan_values_size(const dl_plan_values* values) { return values->value.size(); }
dl_plan_value dl_plan_values_get(const dl_plan_values* values, size_t index) { return {values->value[index].index}; }

dl_plan_compile_options dl_default_plan_compile_options(void) {
    const dl::plan::CompileOptions options;
    return {options.fuse_pointwise};
}
dl_status dl_graph_compile(const dl_graph* graph, const dl_plan_value* outputs, size_t count,
    const dl_plan_compile_options* options, dl_execution_plan** output) {
    return make_handle(output, [&] {
        const auto selection = options ? dl::plan::CompileOptions{options->fuse_pointwise != 0}
            : dl::plan::CompileOptions{};
        return graph->value.compile(values(outputs, count), selection);
    });
}
void dl_execution_plan_release(dl_execution_plan* plan) { delete plan; }
dl_status dl_execution_plan_retain(const dl_execution_plan* plan, dl_execution_plan** output) {
    return make_handle(output, [&] { return plan->value; });
}
dl_plan_info dl_execution_plan_get_info(const dl_execution_plan* plan) {
    const auto& value = plan->value;
    return {value.input_specs().size(), value.output_specs().size(), value.step_count(), value.slot_count(),
        value.fusion_count(), value.storage_lifetimes().size(), value.buffer_requirements().size()};
}
dl_tensor_spec dl_execution_plan_input_spec(const dl_execution_plan* plan, size_t index) {
    return convert(plan->value.input_specs()[index]);
}
dl_tensor_spec dl_execution_plan_output_spec(const dl_execution_plan* plan, size_t index) {
    return convert(plan->value.output_specs()[index]);
}
dl_plan_storage_lifetime dl_execution_plan_storage(const dl_execution_plan* plan, size_t index) {
    const auto& value = plan->value.storage_lifetimes()[index];
    return {{value.origin.index}, value.first_step, value.last_step, value.external};
}
dl_plan_buffer_requirement dl_execution_plan_buffer(const dl_execution_plan* plan, size_t index) {
    const auto& value = plan->value.buffer_requirements()[index];
    return {convert(value.device), value.bytes, value.count};
}
dl_status dl_execution_plan_run(const dl_execution_plan* plan,
    const dl_tensor* const* inputs, size_t count, dl_tensor_list** output) {
    return make_handle(output, [&] { return plan->value(tensor_list(inputs, count)); });
}

dl_status dl_execution_plan_checkpoint(const dl_execution_plan* plan,
    const dl_tensor* const* inputs, size_t count, dl_checkpoint_options options, dl_tensor** output) {
    return make_handle(output, [&] {
        return plan->value.checkpoint(tensor_list(inputs, count),
            {.preserve_rng_state = options.preserve_rng_state != 0});
    });
}

dl_status dl_execution_plan_prepare(const dl_execution_plan* plan, dl_execution_workspace** output) {
    return make_handle(output, [&] { return plan->value.prepare(); });
}
void dl_execution_workspace_release(dl_execution_workspace* workspace) { delete workspace; }
dl_status dl_execution_workspace_run(dl_execution_workspace* workspace,
    const dl_tensor* const* inputs, size_t count, dl_tensor_list** output) {
    return make_handle(output, [&] { return workspace->value(tensor_list(inputs, count)); });
}
dl_status dl_execution_workspace_with_memory(dl_execution_workspace* workspace,
    dl_workspace_fn function, void* user_data) {
    return invoke([&] {
        workspace->value.with_memory([&] {
            const char* error = nullptr;
            if (function(user_data, &error) != DL_OK) {
                throw std::runtime_error(error ? error : "workspace callback failed");
            }
        });
    });
}
dl_workspace_memory_stats dl_execution_workspace_memory_stats(
    const dl_execution_workspace* workspace, dl_device device) {
    const auto value = workspace->value.memory_stats(convert(device));
    return {value.buffers, value.in_use_buffers, value.reserved_bytes, value.in_use_bytes};
}
dl_status dl_execution_workspace_release_unused_memory(dl_execution_workspace* workspace) {
    return invoke([&] { workspace->value.release_unused_memory(); });
}

} // extern "C"
