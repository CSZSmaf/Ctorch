#pragma once

#include "dl/tensor.h"

#include <dlpack/dlpack.h>

namespace dl {

/// 管理生产者提供的描述符和资源；空 deleter 表示生产者保留自行管理的生命周期。
struct DL_API DLPackDeleter {
    void operator()(DLManagedTensorVersioned* tensor) const noexcept;
    void operator()(DLManagedTensor* tensor) const noexcept;
};
using DLPackTensor = std::unique_ptr<DLManagedTensorVersioned, DLPackDeleter>;
using LegacyDLPackTensor = std::unique_ptr<DLManagedTensor, DLPackDeleter>;

/// CUDA 返回框架计算 stream 的整数句柄，供外部生产者建立输入就绪依赖。
/// CPU 返回零，表示该设备不需要 stream 参数。
DL_API std::intptr_t dlpack_consumer_stream(Device device);

/// 共享存储并复制布局元数据，描述符通过 deleter 释放；发布的 ABI 版本为 1.0。
/// CUDA consumer_stream 使用 1=legacy default、2=per-thread default、原生句柄或 -1。
/// 默认使用 legacy default；-1 将就绪与结束依赖交给调用方，CPU 忽略该参数。
/// 消费者的全部使用提交到指定 stream 后释放描述符，stream 保持到释放结束。
/// 使用 2 时，提交和释放在同一宿主线程完成。
/// 外部写入遵循只读标记，并在没有共享读者或待反向保存值时安排。
DL_API DLPackTensor to_dlpack(const Tensor& tensor, std::intptr_t consumer_stream = 1);
/// legacy 格式表达可写存储；只读值先显式 clone 再导出。
DL_API LegacyDLPackTensor to_dlpack_legacy(const Tensor& tensor, std::intptr_t consumer_stream = 1);

/// 接管一个描述符，成功和异常路径均由 RAII 管理其 deleter，原指针不再重复消费。
/// 支持 CPU、CUDA、CUDA pinned host 的六种标量 dtype 和非负 stride。
/// 生产者已将数据安排为在 dlpack_consumer_stream(device) 上可读。
/// 结果共享外部存储，requires_grad=false；外部计算通过 custom_vjp 登记反向。
/// CUDA 外部存储在最后一次框架使用完成后交还生产者，其最终释放可能等待计算。
DL_API Tensor from_dlpack(DLPackTensor tensor);
DL_API Tensor from_dlpack(LegacyDLPackTensor tensor);

} // namespace dl
