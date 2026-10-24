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
    const int64_t shape[] = {3, 2}, channels[] = {2};
    const float values[] = {0, 1, 2, 3, 4, 5};
    const dl_tensor_options options = dl_default_tensor_options();
    dl_tensor *input, *mean, *variance;
    require_ok(dl_from_host(values, (dl_shape){shape, 2}, options, &input));
    require_ok(dl_zeros((dl_shape){channels, 1}, options, &mean));
    require_ok(dl_ones((dl_shape){channels, 1}, options, &variance));

    /* 函数式调用交付新状态，列表释放后，取出的拥有型句柄继续保持数据。 */
    dl_tensor_list* result;
    require_ok(dl_batch_norm(input, mean, variance, NULL, NULL, 1, 0.1, 1e-5, &result));
    dl_tensor *next_mean, *next_variance;
    require_ok(dl_tensor_list_get(result, 1, &next_mean));
    require_ok(dl_tensor_list_get(result, 2, &next_variance));
    dl_tensor_list_release(result);
    float original[2], updated[2];
    require_ok(dl_tensor_copy_to_host(mean, original));
    require_ok(dl_tensor_copy_to_host(next_mean, updated));
    printf("mean before %.7g %.7g, after %.7g %.7g\n",
        (double)original[0], (double)original[1], (double)updated[0], (double)updated[1]);

    /* 下一批直接更新已选定的状态，调用方保持这两个状态句柄。 */
    dl_tensor* output;
    require_ok(dl_batch_norm_update(input, next_mean, next_variance, NULL, NULL, 1, 0.1, 1e-5, &output));
    dl_tensor_release(output);
    dl_tensor_release(next_variance);
    dl_tensor_release(next_mean);
    dl_tensor_release(variance);
    dl_tensor_release(mean);
    dl_tensor_release(input);
}
