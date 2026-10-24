#include <dl/c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

int main(void) {
    const int64_t dimensions[] = {2};
    const dl_shape shape = {dimensions, 1};
    const float initial[] = {3, -2};
    dl_tensor_options tensor_options = dl_default_tensor_options();
    tensor_options.requires_grad = 1;
    dl_tensor *parameter, *velocity = NULL;
    require_ok(dl_from_host(initial, shape, tensor_options, &parameter));
    dl_grad_scaler_options scaling = dl_default_grad_scaler_options();
    scaling.initial_scale = 128.0;
    scaling.growth_interval = 4;
    dl_grad_scaler* scaler;
    require_ok(dl_grad_scaler_create(scaling, &scaler));
    dl_sgd_options sgd = dl_default_sgd_options();
    sgd.learning_rate = 0.1;
    sgd.momentum = 0.9;
    int updates = 0;

    for (int forward = 0; forward < 8; ++forward) {
        dl_tensor *square, *loss, *scaled_loss;
        require_ok(dl_mul(parameter, parameter, &square));
        require_ok(dl_sum_all(square, &loss));
        require_ok(dl_grad_scaler_scale(scaler, loss, &scaled_loss));
        const dl_tensor* inputs[] = {parameter};
        dl_gradients* gradients;
        require_ok(dl_grad(scaled_loss, inputs, 1, NULL, &gradients));
        int32_t finite;
        require_ok(dl_grad_scaler_unscale_gradients_update(scaler, gradients, &finite));
        if (finite) {
            dl_tensor *norm, *gradient;
            require_ok(dl_clip_gradients_norm_update(gradients, 1.0, 2.0, &norm));
            require_ok(dl_gradients_get(gradients, 0, &gradient));
            dl_tensor_list* result;
            require_ok(dl_sgd(parameter, gradient, velocity, sgd, &result));
            dl_tensor *next_parameter, *next_velocity;
            require_ok(dl_tensor_list_get(result, 0, &next_parameter));
            require_ok(dl_tensor_list_get(result, 1, &next_velocity));
            dl_tensor_list_release(result);
            dl_tensor_release(parameter);
            dl_tensor_release(velocity);
            parameter = next_parameter;
            velocity = next_velocity;
            dl_tensor_release(gradient);
            dl_tensor_release(norm);
            ++updates;
        }
        /* 每个窗口只调整一次倍率；跳过更新时参数与动量继续由原句柄持有。 */
        require_ok(dl_grad_scaler_update(scaler, finite));
        printf("forward %d, updates %d, scale %.7g\n", forward + 1, updates,
            dl_grad_scaler_get_scale(scaler));
        dl_gradients_release(gradients);
        dl_tensor_release(scaled_loss);
        dl_tensor_release(loss);
        dl_tensor_release(square);
    }
    dl_grad_scaler_release(scaler);
    dl_tensor_release(velocity);
    dl_tensor_release(parameter);
}
