#include <dl/npu_c_api.h>

#include <stdio.h>
#include <stdlib.h>

static void require_ok(dl_status status) {
    if (status != DL_OK) {
        fprintf(stderr, "%s\n", dl_last_error());
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: ctorch_npu_c_infer model.onnx inputs.ct outputs.ct\n");
        return 1;
    }

    dl_npu_model* model;
    dl_npu_request* request;
    dl_state* inputs;
    require_ok(dl_npu_model_create(argv[1], NULL, &model));
    require_ok(dl_npu_request_create(model, &request));
    require_ok(dl_state_load(argv[2], &inputs));

    /* 文件条目按输入端口排列；请求在 set_input 内完成复制，随后可释放原句柄。 */
    for (size_t i = 0; i < dl_npu_model_input_count(model); ++i) {
        dl_tensor* input;
        require_ok(dl_state_tensor(inputs, i, &input));
        require_ok(dl_npu_request_set_input(request, i, input));
        dl_tensor_release(input);
    }
    dl_state_release(inputs);
    require_ok(dl_npu_request_start_async(request));
    require_ok(dl_npu_request_wait(request));

    /* 输出拥有独立 CPU 存储；状态表复制名称并共享这些张量的存储。 */
    const size_t count = dl_npu_model_output_count(model);
    dl_tensor** tensors = malloc(count * sizeof(*tensors));
    const char** names = malloc(count * sizeof(*names));
    char (*fallback_names)[32] = malloc(count * sizeof(*fallback_names));
    for (size_t i = 0; i < count; ++i) {
        const dl_npu_port port = dl_npu_model_output_port(model, i);
        snprintf(fallback_names[i], sizeof(fallback_names[i]), "output_%zu", i);
        names[i] = port.name[0] ? port.name : fallback_names[i];
        require_ok(dl_npu_request_output(request, i, &tensors[i]));
    }
    dl_state* outputs;
    require_ok(dl_state_create(names, (const dl_tensor* const*)tensors, count, &outputs));
    require_ok(dl_state_save(argv[3], outputs));
    dl_state_release(outputs);
    for (size_t i = 0; i < count; ++i) dl_tensor_release(tensors[i]);
    free(fallback_names);
    free(names);
    free(tensors);
    dl_npu_request_release(request);
    dl_npu_model_release(model);
    printf("Saved %zu output tensors\n", count);
    return 0;
}
