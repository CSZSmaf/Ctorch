#include <dl/c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc > 2) {
        fprintf(stderr, "usage: ctorch_c_train [resume-checkpoint]\n");
        return 1;
    }
    const int64_t input_shape[] = {4, 1}, weight_shape[] = {1, 1}, bias_shape[] = {1};
    const float inputs[] = {-1, 0, 1, 2}, targets[] = {-1, 1, 3, 5};
    dl_tensor_options options = dl_default_tensor_options();
    dl_tensor *input, *target, *weight, *bias;
    require_ok(dl_from_host(inputs, (dl_shape){input_shape, 2}, options, &input));
    require_ok(dl_from_host(targets, (dl_shape){input_shape, 2}, options, &target));
    options.requires_grad = 1;
    require_ok(dl_zeros((dl_shape){weight_shape, 2}, options, &weight));
    require_ok(dl_zeros((dl_shape){bias_shape, 1}, options, &bias));
    dl_module* model;
    require_ok(dl_linear_from_tensors(weight, bias, &model));
    const dl_tensor* parameters[] = {weight, bias};
    dl_sgd_options update = dl_default_sgd_options();
    update.learning_rate = 0.05;
    update.momentum = 0.9;
    dl_optimizer* optimizer;
    require_ok(dl_sgd_create(parameters, 2, update, &optimizer));
    dl_training_progress progress = {0, 0};
    if (argc == 2) {
        dl_checkpoint* checkpoint;
        require_ok(dl_checkpoint_load(argv[1], &checkpoint));
        require_ok(dl_checkpoint_restore(checkpoint, model, optimizer, NULL, NULL, 0, &progress));
        dl_checkpoint_release(checkpoint);
    }

    /* 每次前向创建自己的求导图；反向完成后更新参数，再释放本次结果句柄。 */
    while (progress.step < 200) {
        dl_tensor *prediction, *loss;
        require_ok(dl_optimizer_zero_grad(optimizer));
        require_ok(dl_module_forward(model, input, &prediction));
        require_ok(dl_mse_loss(prediction, target, &loss));
        require_ok(dl_backward(loss, NULL));
        require_ok(dl_optimizer_step(optimizer));
        ++progress.step;
        if (progress.step % 20 == 0) {
            float value;
            require_ok(dl_tensor_copy_to_host(loss, &value));
            printf("step %lld, loss %.7g\n", (long long)progress.step, (double)value);
        }
        dl_tensor_release(loss);
        dl_tensor_release(prediction);
    }

    dl_state* state;
    require_ok(dl_module_state(model, &state));
    require_ok(dl_state_save("linear_c.ct", state));
    dl_state_release(state);
    dl_checkpoint* checkpoint;
    require_ok(dl_checkpoint_capture(model, optimizer, progress, NULL, NULL, 0, &checkpoint));
    require_ok(dl_checkpoint_save("linear_c.train.ct", checkpoint));
    dl_checkpoint_release(checkpoint);
    dl_optimizer_release(optimizer);
    dl_module_release(model);
    dl_tensor_release(weight);
    dl_tensor_release(bias);
    dl_tensor_release(input);
    dl_tensor_release(target);
    return 0;
}
