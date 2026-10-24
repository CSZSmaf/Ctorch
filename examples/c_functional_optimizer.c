#include <dl/c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

/* 带计数的结果以参数开头、CPU int64 标量结尾，中间保留各算法的缓冲。 */
static void show_update(const char* name, const dl_tensor_list* result) {
    dl_tensor *parameter, *counter;
    float values[2];
    int64_t step;
    require_ok(dl_tensor_list_get(result, 0, &parameter));
    require_ok(dl_tensor_list_get(result, dl_tensor_list_size(result) - 1, &counter));
    require_ok(dl_tensor_copy_to_host(parameter, values));
    require_ok(dl_tensor_copy_to_host(counter, &step));
    printf("%s weight %.7g %.7g after %lld updates\n", name,
        (double)values[0], (double)values[1], (long long)step);
    dl_tensor_release(counter);
    dl_tensor_release(parameter);
}

int main(void) {
    const int64_t dimensions[] = {2};
    const dl_shape shape = {dimensions, 1};
    const dl_tensor_options options = dl_default_tensor_options();
    const float values[] = {0.5f, -0.25f};
    dl_tensor_options parameter_options = options;
    parameter_options.requires_grad = 1;
    dl_tensor *parameter, *gradient, *coefficient, *weighted, *loss;
    require_ok(dl_ones(shape, parameter_options, &parameter));
    require_ok(dl_from_host(values, shape, options, &coefficient));
    require_ok(dl_mul(parameter, coefficient, &weighted));
    require_ok(dl_sum_all(weighted, &loss));

    /* 梯度以独立结果传给更新函数，参数的叶梯度槽保持为空。 */
    const dl_tensor* arguments[] = {parameter};
    dl_gradients* gradients;
    require_ok(dl_grad(loss, arguments, 1, NULL, &gradients));
    require_ok(dl_gradients_get(gradients, 0, &gradient));
    dl_gradients_release(gradients);
    dl_tensor_release(loss);
    dl_tensor_release(weighted);
    dl_tensor_release(coefficient);

    /* 函数式 SGD 返回独立参数；后续原地更新继续使用选定的参数和动量。 */
    dl_sgd_options sgd = dl_default_sgd_options();
    sgd.momentum = 0.9;
    dl_tensor_list* result;
    require_ok(dl_sgd(parameter, gradient, NULL, sgd, &result));
    dl_tensor *next, *velocity;
    require_ok(dl_tensor_list_get(result, 0, &next));
    require_ok(dl_tensor_list_get(result, 1, &velocity));
    dl_tensor_list_release(result);
    require_ok(dl_sgd_update(next, gradient, velocity, sgd, &result));
    dl_tensor_list_release(result);
    float updated[2];
    require_ok(dl_tensor_copy_to_host(next, updated));
    printf("SGD weight %.7g %.7g\n", (double)updated[0], (double)updated[1]);
    dl_tensor_release(velocity);
    dl_tensor_release(next);

    /* 矩估计通过普通句柄传递，计数是 CPU int64 标量，单独读入宿主循环。 */
    const dl_adamw_options adamw = dl_default_adamw_options();
    require_ok(dl_adamw(parameter, gradient, NULL, NULL, 0, adamw, &result));
    dl_tensor *first, *second, *counter;
    require_ok(dl_tensor_list_get(result, 0, &next));
    require_ok(dl_tensor_list_get(result, 1, &first));
    require_ok(dl_tensor_list_get(result, 2, &second));
    require_ok(dl_tensor_list_get(result, 3, &counter));
    dl_tensor_list_release(result);
    int64_t step;
    require_ok(dl_tensor_copy_to_host(counter, &step));
    dl_tensor_release(counter);

    require_ok(dl_adamw_update(next, gradient, first, second, step, adamw, &result));
    require_ok(dl_tensor_list_get(result, 3, &counter));
    dl_tensor_list_release(result);
    require_ok(dl_tensor_copy_to_host(counter, &step));
    require_ok(dl_tensor_copy_to_host(next, updated));
    printf("AdamW weight %.7g %.7g after %lld updates\n", (double)updated[0], (double)updated[1],
        (long long)step);
    dl_tensor_release(counter);
    dl_tensor_release(second);
    dl_tensor_release(first);
    dl_tensor_release(next);

    const dl_adam_options adam = dl_default_adam_options();
    require_ok(dl_adam(parameter, gradient, NULL, NULL, 0, adam, &result));
    show_update("Adam", result);
    dl_tensor_list_release(result);

    /* centered 和动量同时开启时，结果顺序为参数、平方均值、动量、梯度均值、计数。 */
    dl_rmsprop_options rmsprop = dl_default_rmsprop_options();
    rmsprop.momentum = 0.9;
    rmsprop.centered = 1;
    require_ok(dl_rmsprop(parameter, gradient, NULL, NULL, NULL, 0, rmsprop, &result));
    dl_tensor *square_average, *gradient_average;
    require_ok(dl_tensor_list_get(result, 0, &next));
    require_ok(dl_tensor_list_get(result, 1, &square_average));
    require_ok(dl_tensor_list_get(result, 2, &velocity));
    require_ok(dl_tensor_list_get(result, 3, &gradient_average));
    require_ok(dl_tensor_list_get(result, 4, &counter));
    require_ok(dl_tensor_copy_to_host(counter, &step));
    dl_tensor_release(counter);
    dl_tensor_list_release(result);
    require_ok(dl_rmsprop_update(next, gradient, square_average, velocity, gradient_average, step, rmsprop, &result));
    show_update("RMSprop", result);
    dl_tensor_list_release(result);
    dl_tensor_release(gradient_average);
    dl_tensor_release(velocity);
    dl_tensor_release(square_average);
    dl_tensor_release(next);

    dl_adagrad_options adagrad = dl_default_adagrad_options();
    adagrad.initial_accumulator = 0.1;
    adagrad.learning_rate_decay = 0.02;
    require_ok(dl_adagrad(parameter, gradient, NULL, 0, adagrad, &result));
    show_update("Adagrad", result);
    dl_tensor_list_release(result);
    dl_tensor_release(gradient);
    dl_tensor_release(parameter);
}
