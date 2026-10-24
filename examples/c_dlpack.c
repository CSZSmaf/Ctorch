#include <dl/dlpack_c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

int main(void) {
    const int64_t shape[] = {2, 3};
    const float values[] = {1, 2, 3, 4, 5, 6};
    dl_tensor *input, *view, *shared;
    DLManagedTensorVersioned* descriptor;
    require_ok(dl_from_host(values, (dl_shape){shape, 2}, dl_default_tensor_options(), &input));
    require_ok(dl_transpose(input, 0, 1, &view));
    require_ok(dl_to_dlpack(view, 0, &descriptor));

    /* 描述符拥有布局并保留底层存储，因此原句柄可以先释放。 */
    dl_tensor_release(view);
    dl_tensor_release(input);
    require_ok(dl_from_dlpack(descriptor, &shared));
    descriptor = NULL; /* 拥有权已经移交给 shared 的存储。 */

    float output[6];
    require_ok(dl_tensor_copy_to_host(shared, output));
    printf("transposed");
    for (size_t i = 0; i < 6; ++i) printf(" %.1f", output[i]);
    printf("\n");
    dl_tensor_release(shared);
    return 0;
}
