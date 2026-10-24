#include <dl/c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

/* 保存前向输出用于 g * exp(x)，保存值和交付结果各自持有张量存储。 */
static dl_status exponential_forward(dl_vjp_context* context, const dl_tensor* const* inputs,
    size_t count, void* user_data, dl_tensor** output, const char** error) {
    (void)count;
    (void)user_data;
    dl_status status = dl_exp(inputs[0], output);
    if (status == DL_OK) {
        const dl_tensor* saved[] = {*output};
        status = dl_vjp_save_for_backward(context, saved, 1);
    }
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

/* saved_tensor 交付拥有型句柄；临时句柄释放后，框架仍持有所需保存值。 */
static dl_status exponential_backward(const dl_vjp_context* context, const dl_tensor* grad,
    void* user_data, dl_tensor** gradients, size_t count, const char** error) {
    (void)count;
    (void)user_data;
    dl_tensor* saved = NULL;
    dl_status status = dl_vjp_saved_tensor(context, 0, &saved);
    if (status == DL_OK) status = dl_mul(grad, saved, &gradients[0]);
    dl_tensor_release(saved);
    if (status != DL_OK) *error = dl_last_error();
    return status;
}

int main(void) {
    const int64_t shape[] = {3}, axes[] = {0};
    const float values[] = {-1, 0, 1};
    dl_tensor_options options = dl_default_tensor_options();
    options.requires_grad = 1;
    dl_tensor *input, *output, *loss, *gradient;
    require_ok(dl_from_host(values, (dl_shape){shape, 1}, options, &input));
    const dl_tensor* inputs[] = {input};
    require_ok(dl_custom_vjp(inputs, 1, exponential_forward, exponential_backward, NULL, NULL, &output));
    require_ok(dl_sum(output, (dl_shape){axes, 1}, 0, &loss));
    require_ok(dl_backward(loss, NULL));
    require_ok(dl_tensor_gradient(input, &gradient));
    float result[3];
    require_ok(dl_tensor_copy_to_host(gradient, result));
    printf("gradient %.6f %.6f %.6f\n", result[0], result[1], result[2]);
    dl_tensor_release(gradient);
    dl_tensor_release(loss);
    dl_tensor_release(output);
    dl_tensor_release(input);
    return 0;
}
