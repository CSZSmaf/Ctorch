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
    require_ok(dl_manual_seed(41));
    const int64_t dimensions[] = {2, 2, 64, 32};
    const dl_shape shape = {dimensions, 4};
    dl_tensor_options options = dl_default_tensor_options();
    options.device = (dl_device){DL_CUDA, 0};
    options.requires_grad = 1;
    dl_tensor *query, *key, *value, *target, *output, *loss;
    require_ok(dl_randn(shape, options, NULL, &query));
    require_ok(dl_randn(shape, options, NULL, &key));
    require_ok(dl_randn(shape, options, NULL, &value));
    options.requires_grad = 0;
    require_ok(dl_zeros(shape, options, &target));

    dl_attention_compute_options attention = dl_default_attention_compute_options();
    attention.backend = DL_ATTENTION_CUDNN;
    attention.causal = 1;
    attention.dropout = 0.1;
    const dl_random_key random = {41, 7};
    dl_autocast_scope* precision;
    require_ok(dl_autocast_begin(DL_CUDA, DL_BF16, 1, &precision));
    /* key 按值决定本次概率 Dropout；返回后由反向节点持有重放所需的标量。 */
    require_ok(dl_scaled_dot_product_attention_ex(query, key, value, NULL, attention, &random, &output));
    dl_autocast_end(precision);
    require_ok(dl_mse_loss(output, target, &loss));
    require_ok(dl_backward(loss, NULL));
    float result;
    require_ok(dl_tensor_copy_to_host(loss, &result));
    printf("attention loss %.7g\n", (double)result);

    dl_tensor_release(loss);
    dl_tensor_release(output);
    dl_tensor_release(target);
    dl_tensor_release(value);
    dl_tensor_release(key);
    dl_tensor_release(query);
    return 0;
}
