#include <dl/plan_c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

/* 此描述接收连续预测，两个结果分别保留原张量及其一维视图。 */
static dl_status infer_views(dl_plan_infer_context* context, const dl_tensor_spec* inputs,
    size_t count, void* user_data, const char** error) {
    (void)count;
    (void)user_data;
    dl_plan_output_spec original = {inputs[0], 0, -1};
    dl_status status = dl_plan_infer_add_output(context, original);
    int64_t elements = 1;
    for (size_t axis = 0; axis < inputs[0].shape.size; ++axis) elements *= inputs[0].shape.data[axis];
    const int64_t shape[] = {elements}, strides[] = {1};
    dl_plan_output_spec flat = {inputs[0], -1, 0};
    flat.tensor.shape = (dl_shape){shape, 1};
    flat.tensor.strides = (dl_shape){strides, 1};
    if (status == DL_OK) status = dl_plan_infer_add_output(context, flat);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

/* 列表复制张量身份，临时视图句柄释放后，交付的列表继续拥有两个结果。 */
static dl_status execute_views(const dl_tensor* const* inputs, size_t count, void* user_data,
    dl_tensor_list** output, const char** error) {
    (void)count;
    (void)user_data;
    const int64_t shape[] = {-1};
    dl_tensor* flat = NULL;
    dl_status status = dl_reshape(inputs[0], (dl_shape){shape, 1}, &flat);
    if (status == DL_OK) {
        const dl_tensor* results[] = {inputs[0], flat};
        status = dl_tensor_list_create(results, 2, output);
    }
    dl_tensor_release(flat);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

static dl_plan_value call_one(dl_graph* graph, const dl_plan_operation* operation,
    const dl_plan_value* inputs, size_t count) {
    dl_plan_values* outputs;
    require_ok(dl_graph_call(graph, operation, inputs, count, &outputs));
    const dl_plan_value value = dl_plan_values_get(outputs, 0);
    dl_plan_values_release(outputs);
    return value;
}

/* 工作区与训练对象由 main 持有；回调把最后一次损失的拥有型句柄保存在 loss。 */
typedef struct TrainingStep {
    dl_execution_workspace* workspace;
    const dl_tensor* const* inputs;
    size_t input_count;
    const dl_tensor* target;
    dl_optimizer* optimizer;
    dl_tensor* loss;
} TrainingStep;

static dl_status train_step(void* user_data, const char** error) {
    TrainingStep* step = user_data;
    dl_tensor_release(step->loss);
    step->loss = NULL;
    dl_tensor_list* predictions = NULL;
    dl_tensor* prediction = NULL;
    dl_status status = dl_optimizer_zero_grad(step->optimizer);
    if (status == DL_OK) status = dl_execution_workspace_run(
        step->workspace, step->inputs, step->input_count, &predictions);
    if (status == DL_OK) status = dl_tensor_list_get(predictions, 0, &prediction);
    dl_tensor_list_release(predictions);
    if (status == DL_OK) status = dl_mse_loss(prediction, step->target, &step->loss);
    if (status == DL_OK) status = dl_backward(step->loss, NULL);
    if (status == DL_OK) status = dl_optimizer_step(step->optimizer);
    dl_tensor_release(prediction);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

int main(void) {
    const int64_t batch_shape[] = {5, 1}, weight_shape[] = {1, 1}, bias_shape[] = {1};
    const float x[] = {-2, -1, 0, 1, 2}, y[] = {0, 0, 1, 3, 5};
    dl_tensor_options options = dl_default_tensor_options();
    dl_tensor *input, *target, *weight, *bias;
    require_ok(dl_from_host(x, (dl_shape){batch_shape, 2}, options, &input));
    require_ok(dl_from_host(y, (dl_shape){batch_shape, 2}, options, &target));
    options.requires_grad = 1;
    require_ok(dl_full((dl_shape){weight_shape, 2}, 0.5, options, &weight));
    require_ok(dl_full((dl_shape){bias_shape, 1}, 0.5, options, &bias));

    dl_graph* graph;
    require_ok(dl_graph_create(&graph));
    dl_plan_value data, w, b;
    require_ok(dl_graph_input(graph, dl_tensor_get_spec(input), "input", &data));
    require_ok(dl_graph_input(graph, dl_tensor_get_spec(weight), "weight", &w));
    require_ok(dl_graph_input(graph, dl_tensor_get_spec(bias), "bias", &b));
    dl_plan_operation *linear, *add, *relu, *views;
    require_ok(dl_plan_builtin_operation(DL_PLAN_LINEAR, &linear));
    require_ok(dl_plan_builtin_operation(DL_PLAN_ADD, &add));
    require_ok(dl_plan_builtin_operation(DL_PLAN_RELU, &relu));
    require_ok(dl_plan_custom_operation("prediction_views", infer_views, execute_views,
        DL_PLAN_PURE, NULL, NULL, &views));
    dl_plan_value result = call_one(graph, linear, (dl_plan_value[]){data, w}, 2);
    result = call_one(graph, add, (dl_plan_value[]){result, b}, 2);
    result = call_one(graph, relu, &result, 1);
    dl_plan_values* results;
    require_ok(dl_graph_call(graph, views, &result, 1, &results));
    const dl_plan_value outputs[] = {dl_plan_values_get(results, 0), dl_plan_values_get(results, 1)};
    dl_execution_plan* plan;
    require_ok(dl_graph_compile(graph, outputs, 2, NULL, &plan));
    dl_plan_values_release(results);
    dl_plan_operation_release(views);
    dl_plan_operation_release(relu);
    dl_plan_operation_release(add);
    dl_plan_operation_release(linear);
    dl_graph_release(graph);

    const dl_plan_info info = dl_execution_plan_get_info(plan);
    printf("steps %zu, fusions %zu, buffer classes %zu\n", info.step_count, info.fusion_count, info.buffer_count);
    dl_execution_workspace* workspace;
    require_ok(dl_execution_plan_prepare(plan, &workspace));
    /* 工作区继续持有不可变计划，原建图和计划句柄已经完成交接。 */
    dl_execution_plan_release(plan);
    const dl_tensor* parameters[] = {weight, bias};
    dl_sgd_options update = dl_default_sgd_options();
    update.learning_rate = 0.05;
    dl_optimizer* optimizer;
    require_ok(dl_sgd_create(parameters, 2, update, &optimizer));
    const dl_tensor* bindings[] = {input, weight, bias};
    TrainingStep training = {.workspace = workspace, .inputs = bindings, .input_count = 3,
        .target = target, .optimizer = optimizer};
    for (int step = 0; step < 40; ++step) {
        require_ok(dl_execution_workspace_with_memory(workspace, train_step, &training));
    }
    /* 损失句柄跨作用域存活，在训练结束后集中读取一次宿主标量。 */
    float last_loss;
    require_ok(dl_tensor_copy_to_host(training.loss, &last_loss));
    dl_tensor_release(training.loss);
    printf("last training loss %.7g\n", (double)last_loss);
    dl_workspace_memory_stats memory = dl_execution_workspace_memory_stats(workspace, options.device);
    printf("reserved bytes %zu, in use %zu\n", memory.reserved_bytes, memory.in_use_bytes);
    require_ok(dl_execution_workspace_release_unused_memory(workspace));
    dl_execution_workspace_release(workspace);
    dl_optimizer_release(optimizer);
    dl_tensor_release(bias);
    dl_tensor_release(weight);
    dl_tensor_release(target);
    dl_tensor_release(input);
    return 0;
}
