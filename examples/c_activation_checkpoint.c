#include <dl/plan_c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

/* 回调只使用本次输入句柄；同一段普通算子同时承担初次前向和反向时的重算。 */
static dl_status block(const dl_tensor* const* inputs, size_t count, void* user_data,
    dl_tensor** output, const char** error) {
    (void)count;
    (void)user_data;
    dl_tensor *projection = NULL, *shifted = NULL, *activated = NULL;
    dl_status status = dl_matmul(inputs[0], inputs[1], &projection);
    if (status == DL_OK) status = dl_add(projection, inputs[2], &shifted);
    if (status == DL_OK) status = dl_relu(shifted, &activated);
    if (status == DL_OK) status = dl_dropout(activated, 0.1, 1, output);
    dl_tensor_release(activated);
    dl_tensor_release(shifted);
    dl_tensor_release(projection);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

/* 此示例的 float32 矩阵经线性投影得到 [batch, channels]，偏置与激活保持形状。 */
static dl_status block_spec(dl_plan_infer_context* context, const dl_tensor_spec* inputs,
    size_t count, void* user_data, const char** error) {
    (void)count;
    (void)user_data;
    const int64_t shape[] = {inputs[0].shape.data[0], inputs[1].shape.data[1]};
    dl_status status = dl_plan_infer_contiguous_output(context, (dl_shape){shape, 2},
        inputs[0].dtype, inputs[0].device);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

/* 计划回调以列表交付结果，普通重算回调使用单个张量；两者共用上面的计算函数。 */
static dl_status block_values(const dl_tensor* const* inputs, size_t count, void* user_data,
    dl_tensor_list** output, const char** error) {
    dl_tensor* result = NULL;
    dl_status status = block(inputs, count, user_data, &result, error);
    const dl_tensor* values[] = {result};
    if (status == DL_OK) status = dl_tensor_list_create(values, 1, output);
    dl_tensor_release(result);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

int main(void) {
    require_ok(dl_manual_seed(41));
    const int64_t data_shape[] = {5, 1}, weight_shape[] = {1, 1}, bias_shape[] = {1};
    const float x[] = {-2, -1, 0, 1, 2}, y[] = {0, 0, 1, 3, 5};
    dl_tensor_options options = dl_default_tensor_options();
    dl_tensor *input, *target, *weight, *bias;
    require_ok(dl_from_host(x, (dl_shape){data_shape, 2}, options, &input));
    require_ok(dl_from_host(y, (dl_shape){data_shape, 2}, options, &target));
    options.requires_grad = 1;
    require_ok(dl_full((dl_shape){weight_shape, 2}, 0.5, options, &weight));
    require_ok(dl_full((dl_shape){bias_shape, 1}, 0.5, options, &bias));
    const dl_tensor* parameters[] = {weight, bias};
    dl_sgd_options update = dl_default_sgd_options();
    update.learning_rate = 0.05;
    dl_optimizer* optimizer;
    require_ok(dl_sgd_create(parameters, 2, update, &optimizer));

    const dl_tensor* inputs[] = {input, weight, bias};
    dl_graph* graph;
    require_ok(dl_graph_create(&graph));
    dl_plan_value slots[3];
    for (size_t i = 0; i < 3; ++i)
        require_ok(dl_graph_input(graph, dl_tensor_get_spec(inputs[i]), NULL, &slots[i]));
    dl_plan_operation* operation;
    require_ok(dl_plan_custom_operation("hidden", block_spec, block_values, DL_PLAN_RANDOM,
        NULL, NULL, &operation));
    dl_plan_values* results;
    require_ok(dl_graph_call(graph, operation, slots, 3, &results));
    const dl_plan_value hidden = dl_plan_values_get(results, 0);
    dl_execution_plan* plan;
    require_ok(dl_graph_compile(graph, &hidden, 1, NULL, &plan));
    dl_plan_values_release(results);
    dl_plan_operation_release(operation);
    dl_graph_release(graph);

    const dl_checkpoint_options checkpoint = dl_default_checkpoint_options();
    for (int step = 0; step < 40; ++step) {
        require_ok(dl_optimizer_zero_grad(optimizer));
        dl_tensor *prediction, *loss;
        require_ok(dl_execution_plan_checkpoint(plan, inputs, 3, checkpoint, &prediction));
        require_ok(dl_mse_loss(prediction, target, &loss));
        require_ok(dl_backward(loss, NULL));
        require_ok(dl_optimizer_step(optimizer));
        if (step == 39) {
            float value;
            require_ok(dl_tensor_copy_to_host(loss, &value));
            printf("last training loss %.7g\n", (double)value);
        }
        dl_tensor_release(loss);
        dl_tensor_release(prediction);
    }
    dl_execution_plan_release(plan);
    dl_optimizer_release(optimizer);
    dl_tensor_release(bias);
    dl_tensor_release(weight);
    dl_tensor_release(target);
    dl_tensor_release(input);
    return 0;
}
