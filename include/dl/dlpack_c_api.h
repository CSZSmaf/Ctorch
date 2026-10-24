#ifndef CTORCH_DLPACK_C_API_H
#define CTORCH_DLPACK_C_API_H

#include "dl/c_api.h"
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU 返回零，CUDA 返回本框架的计算 stream，传给外部生产者后再导入其结果。 */
DL_API dl_status dl_dlpack_consumer_stream(dl_device device, intptr_t* stream);
/* 导出共享存储；CUDA stream 使用 1、2、原生句柄或 -1，CPU 忽略该参数。
 * 消费提交到指定 stream 后释放描述符；2 的提交与释放使用同一宿主线程。
 * -1 由调用方完成就绪和结束同步。输出的 shape/strides 由描述符自身持有。 */
DL_API dl_status dl_to_dlpack(const dl_tensor* tensor, intptr_t consumer_stream,
    DLManagedTensorVersioned** output);
DL_API dl_status dl_to_dlpack_legacy(const dl_tensor* tensor, intptr_t consumer_stream,
    DLManagedTensor** output);
/* 从调用开始接管描述符，失败时也调用其 deleter；输出句柄在失败时为 NULL。
 * 数据已经在本框架的计算 stream 上就绪，结果 requires_grad=false。 */
DL_API dl_status dl_from_dlpack(DLManagedTensorVersioned* tensor, dl_tensor** output);
DL_API dl_status dl_from_dlpack_legacy(DLManagedTensor* tensor, dl_tensor** output);
/* 尚未移交给消费者的描述符由这两个入口释放；它们也接受 NULL。 */
DL_API void dl_dlpack_release(DLManagedTensorVersioned* tensor);
DL_API void dl_dlpack_release_legacy(DLManagedTensor* tensor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CTORCH_DLPACK_C_API_H
