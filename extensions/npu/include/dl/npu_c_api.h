#ifndef CTORCH_NPU_C_API_H
#define CTORCH_NPU_C_API_H

#include "dl/c_api.h"
#include "dl/npu_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl_npu_model dl_npu_model;
typedef struct dl_npu_request dl_npu_request;

typedef enum dl_npu_performance {
    DL_NPU_LATENCY = 0, DL_NPU_THROUGHPUT = 1
} dl_npu_performance;

/* 配置只在模型创建期间借用，字符串使用 UTF-8。
 * cache_directory 为 NULL 或空串时采用运行时默认缓存策略。
 * input_shape_count 为零时保留模型形状，否则按端口顺序给出全部固定输入形状。 */
typedef struct dl_npu_compile_options {
    dl_npu_performance performance;
    const char* cache_directory;
    const dl_shape* input_shapes;
    size_t input_shape_count;
} dl_npu_compile_options;

/* 名称和形状借用模型的元信息；保留模型句柄期间可重复使用这些地址。 */
typedef struct dl_npu_port {
    const char* name;
    dl_dtype dtype;
    dl_shape shape;
} dl_npu_port;

/* NPU 状态入口共用 dl_last_error()，新句柄的输出槽在失败时为 NULL。
 * 创建和 retain 交付拥有型句柄，release 接受 NULL。 */
DL_NPU_API dl_npu_compile_options dl_npu_default_compile_options(void);
/* options 为 NULL 时使用默认配置；path 指向 ONNX 或 OpenVINO IR。 */
DL_NPU_API dl_status dl_npu_model_create(const char* path, const dl_npu_compile_options* options,
    dl_npu_model** output);
DL_NPU_API dl_status dl_npu_model_retain(const dl_npu_model* model, dl_npu_model** output);
DL_NPU_API void dl_npu_model_release(dl_npu_model* model);
DL_NPU_API size_t dl_npu_model_input_count(const dl_npu_model* model);
DL_NPU_API size_t dl_npu_model_output_count(const dl_npu_model* model);
DL_NPU_API dl_npu_port dl_npu_model_input_port(const dl_npu_model* model, size_t index);
DL_NPU_API dl_npu_port dl_npu_model_output_port(const dl_npu_model* model, size_t index);
DL_NPU_API dl_status dl_npu_model_optimal_request_count(const dl_npu_model* model, size_t* count);

/* 每个请求保留编译模型，并独占自己的缓冲和状态；不同请求可并行执行。
 * 同一请求依次设置全部输入、执行、读取输出，完成后再开始下一次使用。 */
DL_NPU_API dl_status dl_npu_request_create(const dl_npu_model* model, dl_npu_request** output);
DL_NPU_API void dl_npu_request_release(dl_npu_request* request);
/* 输入匹配端口的 dtype 和 shape；复制完成后，原张量可释放或更新。 */
DL_NPU_API dl_status dl_npu_request_set_input(dl_npu_request* request, size_t index, const dl_tensor* input);
DL_NPU_API dl_status dl_npu_request_infer(dl_npu_request* request);
DL_NPU_API dl_status dl_npu_request_start_async(dl_npu_request* request);
DL_NPU_API dl_status dl_npu_request_wait(dl_npu_request* request);
/* ready 查询已提交的异步请求；DL_OK 且 ready 非零表示结果可读取。 */
DL_NPU_API dl_status dl_npu_request_ready(dl_npu_request* request, int32_t* ready);
/* 完成后返回独立 CPU 张量，使用 dl_tensor_release 释放，后续推理不会覆盖它。 */
DL_NPU_API dl_status dl_npu_request_output(const dl_npu_request* request, size_t index, dl_tensor** output);
/* 完成后写入已有的连续、可写 CPU 张量；dtype、shape 匹配输出端口且 requires_grad 为零。
 * 保留 destination 的句柄和存储，同一个缓冲可重复接收后续推理结果。 */
DL_NPU_API dl_status dl_npu_request_output_into(const dl_npu_request* request, size_t index, dl_tensor* destination);
DL_NPU_API dl_status dl_npu_request_reset_state(dl_npu_request* request);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_NPU_C_API_H
