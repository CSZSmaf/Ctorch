# 组合与执行

## 显式状态

函数式模块将参数、buffer 和训练模式交给调用方管理，计算后交付下一份状态。函数式优化器接收参数、梯度与更新状态，再返回参数和状态结果；它们复用普通模块和优化器的数学实现。

这种组织方式便于表达分支训练、状态快照与固定计划。状态表保存张量引用，何时加载回模块由调用方决定。

示例：[函数式模块](../examples/functional_module.cpp)、[优化器](../examples/functional_optimizer.cpp)、[BatchNorm 状态](../examples/functional_state.cpp)。

## 随机 key

显式 `RandomKey` 以值描述随机流。随机调用不推进外部 Generator，可按训练步、层或分支派生 key，使计算与随机输入一同重放。普通 Generator 继续支持有状态采样。

见 [random_key.cpp](../examples/random_key.cpp)。

## 固定执行计划

普通 eager 训练也可以用 `dl::MemoryGuard` 在循环中复用缓冲，见[性能与内存](performance.md)。

计划通过公开算子规格描述稳定的计算结构，编译时确定布局、依赖、别名和状态顺序。执行使用已有数值与一阶求导入口，支持多输出以及在图内保留共享参数关系。

纯计算中没有被使用的结果可被删除；满足单消费者条件的 add 与 ReLU 可融合为一个带独立 VJP 的算子。计划保留显式输出与状态语义。

工作区预留可复用缓冲。输出和反向保存值仍通过 Storage 所有权存活，工作区结束不提前释放它们。训练内存作用域可覆盖前向、反向和参数更新。

见 [execution_plan.cpp](../examples/execution_plan.cpp)、[多输出计划](../examples/multi_output_plan.cpp)与 [C17 计划](../examples/c_execution_plan.c)。

## 激活重算

`checkpoint` 在前向后释放块内中间激活，反向时重放相同计算并交回输入梯度。输入和参与求导的参数显式传入；随机状态与精度作用域跟随重放。持久状态更新安排在重算块外。

重算通过额外计算换取较低的激活占用，适合按模型结构选择块边界。它也可用于单输出固定计划，见 [activation_checkpoint.cpp](../examples/activation_checkpoint.cpp)。

## CUDA 内存

CUDA 上下文管理计算 stream、复制 stream 和专用 Runtime 内存池。分配、事件依赖与异步归还沿 stream 排序；cuDNN 工作区按需求复用。空闲显存默认跨同步点保留，后续训练步可以继续使用。公开统计描述本框架内存池，不包含设备库的私有分配。

阶段结束时可查询池内使用量、设置保留阈值或回收空闲存储，见 [cuda_memory.cpp](../examples/cuda_memory.cpp)。
