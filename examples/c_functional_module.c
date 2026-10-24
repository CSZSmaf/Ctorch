#include <dl/plan_c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

static dl_optimizer* make_optimizer(const dl_functional_state* state) {
    dl_tensor_list* list;
    require_ok(dl_functional_state_parameters(state, &list));
    const size_t count = dl_tensor_list_size(list);
    dl_tensor** parameters = malloc(count * sizeof(*parameters));
    /* 优化器以相反顺序登记同一组叶；快照按身份关联它们的状态与参数组。 */
    for (size_t i = 0; i < count; ++i) require_ok(dl_tensor_list_get(list, count - 1 - i, &parameters[i]));
    dl_sgd_options options = dl_default_sgd_options();
    options.learning_rate = 0.02;
    options.momentum = 0.9;
    dl_optimizer* optimizer;
    require_ok(dl_sgd_create((const dl_tensor* const*)parameters, count, options, &optimizer));
    for (size_t i = 0; i < count; ++i) dl_tensor_release(parameters[i]);
    free(parameters);
    dl_tensor_list_release(list);
    return optimizer;
}

static void train_until(const dl_functional* functional, dl_functional_state** state, dl_optimizer* optimizer,
    const dl_tensor* input, const dl_tensor* target, dl_random_key root, int64_t begin, int64_t end) {
    for (int64_t step = begin; step < end; ++step) {
        require_ok(dl_optimizer_zero_grad(optimizer));
        dl_tensor *prediction, *loss;
        dl_functional_state* next;
        require_ok(dl_functional_call(functional, *state, input, dl_random_fold_in(root, (uint64_t)step),
            &prediction, &next));
        /* 后继状态共用参数身份，原地优化器继续更新这一组叶，buffer 则独立交接。 */
        dl_functional_state_release(*state);
        *state = next;
        require_ok(dl_mse_loss(prediction, target, &loss));
        require_ok(dl_backward(loss, NULL));
        require_ok(dl_optimizer_step(optimizer));
        float value;
        require_ok(dl_tensor_copy_to_host(loss, &value));
        printf("step %lld, loss %.7g\n", (long long)(step + 1), (double)value);
        dl_tensor_release(loss);
        dl_tensor_release(prediction);
    }
}

/* 本例的 Linear、BatchNorm 和 Dropout 保持 [batch, 2]，结果为连续 FP32。 */
static dl_status infer_model(dl_plan_infer_context* context, const dl_tensor_spec* inputs,
    size_t count, void* user_data, const char** error) {
    (void)count;
    (void)user_data;
    dl_status status = dl_plan_infer_contiguous_output(context, inputs[0].shape, DL_F32, inputs[0].device);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

static void evaluate_plan(const dl_functional* functional, const dl_functional_state* state,
    dl_tensor* input, const dl_tensor* target, dl_random_key key) {
    dl_tensor_list *parameters, *buffers;
    require_ok(dl_functional_state_parameters(state, &parameters));
    require_ok(dl_functional_state_buffers(state, &buffers));
    const size_t parameter_count = dl_tensor_list_size(parameters), buffer_count = dl_tensor_list_size(buffers);
    const size_t count = 2 + parameter_count + buffer_count;
    dl_tensor** bindings = malloc(count * sizeof(*bindings));
    bindings[0] = input;
    for (size_t i = 0; i < parameter_count; ++i)
        require_ok(dl_tensor_list_get(parameters, i, &bindings[1 + i]));
    for (size_t i = 0; i < buffer_count; ++i)
        require_ok(dl_tensor_list_get(buffers, i, &bindings[1 + parameter_count + i]));
    require_ok(dl_random_key_tensor(key, &bindings[count - 1]));
    dl_tensor_list_release(buffers);
    dl_tensor_list_release(parameters);

    const size_t module_count = dl_functional_module_count(functional);
    uint8_t* modes = malloc(module_count * sizeof(*modes));
    for (size_t i = 0; i < module_count; ++i) modes[i] = (uint8_t)dl_functional_state_training(state, i);
    dl_plan_operation* operation;
    require_ok(dl_plan_functional_module(functional, modes, module_count, infer_model, NULL, NULL, &operation));
    free(modes);
    dl_graph* graph;
    require_ok(dl_graph_create(&graph));
    dl_plan_value* slots = malloc(count * sizeof(*slots));
    for (size_t i = 0; i < count; ++i)
        require_ok(dl_graph_input(graph, dl_tensor_get_spec(bindings[i]), NULL, &slots[i]));
    dl_plan_values* values;
    require_ok(dl_graph_call(graph, operation, slots, count, &values));
    /* eval 只交付模型结果；节点的前 buffer_count 项是可单独保留的后继 buffer。 */
    const dl_plan_value prediction_slot = dl_plan_values_get(values, buffer_count);
    dl_execution_plan* plan;
    require_ok(dl_graph_compile(graph, &prediction_slot, 1, NULL, &plan));
    dl_plan_values_release(values);
    free(slots);
    dl_graph_release(graph);
    dl_plan_operation_release(operation);

    dl_grad_scope* no_grad;
    require_ok(dl_no_grad_begin(&no_grad));
    dl_tensor_list* results;
    require_ok(dl_execution_plan_run(plan, (const dl_tensor* const*)bindings, count, &results));
    dl_tensor *prediction, *loss;
    require_ok(dl_tensor_list_get(results, 0, &prediction));
    require_ok(dl_mse_loss(prediction, target, &loss));
    float value;
    require_ok(dl_tensor_copy_to_host(loss, &value));
    printf("planned eval loss %.7g\n", (double)value);
    dl_tensor_release(loss);
    dl_tensor_release(prediction);
    dl_tensor_list_release(results);
    dl_no_grad_end(no_grad);
    dl_execution_plan_release(plan);
    for (size_t i = 1; i < count; ++i) dl_tensor_release(bindings[i]);
    free(bindings);
}

int main(void) {
    const dl_tensor_options options = dl_default_tensor_options();
    dl_module *model, *linear, *norm, *dropout;
    require_ok(dl_sequential_create(&model));
    require_ok(dl_linear_create(2, 2, 1, options, &linear));
    require_ok(dl_batch_norm1d_create(2, dl_default_batch_norm_options(), options, &norm));
    require_ok(dl_dropout_create(0.2, &dropout));
    require_ok(dl_sequential_append(model, "linear", linear));
    require_ok(dl_sequential_append(model, "norm", norm));
    require_ok(dl_sequential_append(model, "dropout", dropout));
    dl_module_release(dropout);
    dl_module_release(norm);
    dl_module_release(linear);

    dl_functional* functional;
    require_ok(dl_functional_create(model, &functional));
    dl_module_release(model);
    dl_functional_state* state;
    require_ok(dl_functional_state_create(functional, &state));
    dl_optimizer* optimizer = make_optimizer(state);
    const int64_t dimensions[] = {4, 2};
    const float samples[] = {1, 0, 0, 1, 1, 1, 2, -1};
    const float targets[] = {0.5f, 0, 0, 0.5f, 0.5f, 0.5f, 1, -0.5f};
    dl_tensor *input, *target;
    require_ok(dl_from_host(samples, (dl_shape){dimensions, 2}, options, &input));
    require_ok(dl_from_host(targets, (dl_shape){dimensions, 2}, options, &target));
    dl_random_key root = {73, 0};
    train_until(functional, &state, optimizer, input, target, root, 0, 6);

    dl_checkpoint* checkpoint;
    require_ok(dl_checkpoint_capture_functional(functional, state, optimizer, root,
        (dl_training_progress){0, 6}, NULL, NULL, 0, &checkpoint));
    dl_functional_state* restored;
    dl_optimizer* next_optimizer;
    dl_training_progress progress;
    require_ok(dl_checkpoint_restore_functional(checkpoint, functional, options.device, NULL, NULL, 0,
        &restored, &next_optimizer, &root, &progress));
    dl_checkpoint_release(checkpoint);
    dl_optimizer_release(optimizer);
    dl_functional_state_release(state);
    optimizer = next_optimizer;
    state = restored;
    train_until(functional, &state, optimizer, input, target, root, progress.step, 12);

    require_ok(dl_functional_state_train(state, 0));
    evaluate_plan(functional, state, input, target, root);

    dl_tensor_release(target);
    dl_tensor_release(input);
    dl_optimizer_release(optimizer);
    dl_functional_state_release(state);
    dl_functional_release(functional);
}
