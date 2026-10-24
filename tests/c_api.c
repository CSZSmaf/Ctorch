#include <dl/c_api.h>
#include <math.h>
#include <stdio.h>

#define OK(call) do { if ((call) != DL_OK) { fprintf(stderr, "%s\n", dl_last_error()); return 1; } } while (0)

int main(void) {
    const int64_t dims[] = {3};
    const float values[] = {1, 2, 3};
    dl_tensor_options options = dl_default_tensor_options();
    options.requires_grad = 1;
    dl_tensor *input, *square, *seed, *gradient;
    OK(dl_from_host(values, (dl_shape){dims, 1}, options, &input));
    OK(dl_mul(input, input, &square));
    options.requires_grad = 0;
    OK(dl_ones((dl_shape){dims, 1}, options, &seed));
    OK(dl_backward(square, seed));
    OK(dl_tensor_gradient(input, &gradient));
    float result[3];
    OK(dl_tensor_copy_to_host(gradient, result));
    for (int i = 0; i < 3; ++i) if (fabsf(result[i] - 2 * values[i]) > 1e-6f) return 1;
    dl_tensor_release(gradient);
    dl_tensor_release(seed);
    dl_tensor_release(square);
    dl_tensor_release(input);

    dl_module *model, *layer;
    OK(dl_sequential_create(&model));
    OK(dl_relu_create(&layer));
    OK(dl_sequential_append(model, "activation", layer));
    if (dl_sequential_append(model, "activation", layer) == DL_OK) return 1;
    if (dl_last_error()[0] == '\0') return 1;
    dl_module_release(layer);
    dl_module_release(model);
    return 0;
}
