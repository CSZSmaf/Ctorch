#pragma once

#include "dl/onnx_export.h"
#include "dl/plan_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dl_onnx_binding { dl_plan_value input; const dl_tensor* value; } dl_onnx_binding;
/* path 为 UTF-8 路径；weights 将相应图输入固定为常量，其余输入保留名称。 */
DL_ONNX_API dl_status dl_onnx_save(const char* path, const dl_graph* graph,
    const dl_plan_value* outputs, size_t count, const dl_onnx_binding* weights, size_t weight_count);

#ifdef __cplusplus
}
#endif
