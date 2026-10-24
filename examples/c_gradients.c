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
    const int64_t dimensions[] = {3};
    const dl_shape shape = {dimensions, 1};
    const float values[] = {1, 2, 3};
    dl_tensor_options options = dl_default_tensor_options();
    options.requires_grad = 1;
    dl_tensor *x, *unused, *square, *scaled, *first_seed, *second_seed;
    require_ok(dl_from_host(values, shape, options, &x));
    require_ok(dl_ones(shape, options, &unused));
    require_ok(dl_mul(x, x, &square));
    require_ok(dl_mul_scalar(square, 3.0, &scaled));
    options.requires_grad = 0;
    require_ok(dl_ones(shape, options, &first_seed));
    require_ok(dl_full(shape, 2.0, options, &second_seed));

    /* 中间结果和叶在同一次反向中取值，重复输入保持请求顺序，空项仍占据一个槽。 */
    const dl_tensor* outputs[] = {square, scaled};
    const dl_tensor* inputs[] = {square, x, unused, x};
    const dl_tensor* seeds[] = {first_seed, second_seed};
    const char* names[] = {"square", "x", "unused", "x again"};
    dl_gradients* gradients;
    require_ok(dl_grad_many(outputs, 2, inputs, 4, seeds, &gradients));
    for (size_t i = 0; i < dl_gradients_size(gradients); ++i) {
        dl_tensor* value;
        require_ok(dl_gradients_get(gradients, i, &value));
        printf("%s:", names[i]);
        if (value) {
            float host[3];
            require_ok(dl_tensor_copy_to_host(value, host));
            for (size_t j = 0; j < 3; ++j) printf(" %.7g", (double)host[j]);
            dl_tensor_release(value);
        } else {
            printf(" no contribution");
        }
        printf("\n");
    }
    dl_gradients_release(gradients);
    dl_tensor_release(second_seed);
    dl_tensor_release(first_seed);
    dl_tensor_release(scaled);
    dl_tensor_release(square);
    dl_tensor_release(unused);
    dl_tensor_release(x);
}
