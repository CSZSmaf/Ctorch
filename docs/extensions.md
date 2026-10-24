# C17 与外部计算

## C17

`<dl/c_api.h>` 提供拥有型 opaque handle。创建、运算和返回梯度的入口通过输出参数交付句柄，调用方使用相应的 release 函数结束拥有权。shape、stride 和 data 视图借用所属张量的生命周期。

可能失败的入口返回 `dl_status`，错误文字由 `dl_last_error()` 提供。统一边界将 C++ 异常转换成状态码；C 包装复用主库的张量、求导与训练引擎。

示例见 [c_train.c](../examples/c_train.c)、[C17 函数式模块](../examples/c_functional_module.c)与 [C17 优化器](../examples/c_functional_optimizer.c)。

## 自定义 VJP

`<dl/extension.h>` 为外部计算提供一阶 VJP 接口。前向接收输入并通过上下文保存反向需要的数据；反向按输入顺序交付梯度贡献。只保存公式实际需要的张量或元信息。

C17 使用对应回调及上下文拥有权接口。固定计划可配合规格推导函数描述外部算子的结果布局。完整示例见 [custom_vjp.cpp](../examples/custom_vjp.cpp)和 [c_custom_vjp.c](../examples/c_custom_vjp.c)。

## DLPack

启用 `CTORCH_USE_DLPACK` 后，`<dl/dlpack.h>` 与 `<dl/dlpack_c_api.h>` 提供官方 DLPack 1.x ABI 的张量交换。

| 接口 | 作用 |
| --- | --- |
| to_dlpack | 导出拥有型版本化描述符，输出版本为 1.0 |
| to_dlpack_legacy | 导出旧式 DLManagedTensor |
| from_dlpack | 接管描述符，建立共享存储且不带求导历史的张量 |
| dlpack_consumer_stream | 查询框架的 CUDA 消费 stream；CPU 返回零 |

CPU、CUDA 和页锁定 CPU 存储使用相应设备类型。交换保留 shape、非负 stride 与 byte offset，支持 f32、f64、i64、bool、f16、bf16。新版描述符传播只读属性；旧式描述符不能表示只读存储，可先显式 clone。

描述符保留布局和存储，原张量句柄可以先释放。导入从调用开始接管拥有权，即使导入失败也负责调用 producer deleter；同一描述符只转交一次。尚未移交的 C 描述符使用 `dl_dlpack_release` 或对应 legacy 入口释放。

### CUDA 交接

导入前，producer 将数据安排为在 `dlpack_consumer_stream(device)` 上可读。导出接收 consumer stream：`1` 表示 legacy default，`2` 表示 per-thread default，其他值可为原生句柄；`-1` 表示调用方自行管理同步。

正常导出使用事件传递就绪依赖。consumer 向声明的 stream 提交全部消费后释放描述符，释放路径把结束依赖传回框架 stream，再归还存储。consumer stream 保持存活至描述符释放结束，per-thread default 的提交与释放使用同一宿主线程。

外部 CUDA 存储的最后一次释放会等待框架计算完成，再调用 producer deleter。CPU 数据在交接时已完成写入。设备转换、精度转换和跨框架求导由应用显式组织。

示例见 [DLPack 外部算子与 VJP](../examples/dlpack_vjp.cpp)和 [C17 共享视图](../examples/c_dlpack.c)。主库不提供 Python capsule 绑定。
