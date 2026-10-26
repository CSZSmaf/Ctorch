# 张量与梯度

## 类型与设备

CPU 通用浮点计算使用 `f32` / `f64`，CUDA 通用浮点计算使用 `f32`。`f16` / `bf16` 用于低精度存储及选定的矩阵乘、卷积和注意力路径；归约与主参数按算子约定使用 FP32。索引使用 `i64`，掩码使用 `bool`。

存储类型、计算类型和输出类型可以不同。例如，BF16 输入的矩阵乘使用 FP32 累加，并返回 FP32 张量。

| 设备与操作 | 输入存储 | 计算与累加 | 输出 |
| --- | --- | --- | --- |
| CPU 通用浮点计算 | FP32 / FP64 | 对应浮点类型 | 对应浮点类型 |
| CPU sum、mean、var、stddev | FP16 / BF16 / FP32 / FP64 | FP64 统计累加 | FP64 输入返回 FP64，其余返回 FP32 |
| CPU LayerNorm、RMSNorm、BatchNorm | FP16 / BF16 / FP32 / FP64 | FP64 统计及输入梯度归约 | FP64 输入返回 FP64，其余返回 FP32 |
| CPU 低精度矩阵乘 | FP16 / BF16 | 转成 FP32 计算 | FP32 |
| CUDA 通用浮点计算 | FP32 | FP32 | FP32 |
| CUDA 低精度逐元素运算与归一化 | FP16 / BF16 | 读取后转成 FP32 计算 | FP32 |
| CUDA 低精度矩阵乘 | FP16 / BF16 | FP32 累加 | FP32 |
| cuDNN 低精度卷积 | FP16 / BF16 | 低精度乘法、FP32 累加，按选定低精度写回 | 展开为 FP32 |
| cuDNN 注意力 | FP16 / BF16 | 由融合注意力内核组织低精度计算和归约 | FP32 |

CPU 统计保留宽精度的均值，再计算中心平方和。输出张量的类型保持不变。CUDA 的归约和归一化使用 FP32 分块计算。低精度量化发生在读取之前，统计累加无法恢复量化时已经丢失的信息。

FP64 数据可以保存、迁移和交换；CUDA 通用运算采用上表中的类型。`astype()` 显式转换类型，`AutocastGuard` 选择矩阵乘、卷积等算子的计算输入类型。CPU 的 FP16/BF16 矩阵乘使用 FP32 内核。

通过 `TensorOptions::device` 选择设备，或调用 `Tensor::to(Device)` 显式迁移。一次运算的输入使用相同设备及对应算子要求的 dtype。CUDA 内核布局支持至多 16 维。

## 读取数值

`item<T>()` 将单元素张量转换为 `T` 的数值。`to_vector<T>()` 按逻辑行主序转换并复制全部元素，转置和切片也按逻辑顺序读取。

```cpp
auto x = dl::full({}, 1.5, {.dtype = dl::DType::f64});
float value = x.item<float>();                 // 1.5f
auto values = x.to_vector<float>();            // {1.5f}
auto index = x.item<dl::Index>();               // 1
```

`T` 可用 `float`、`double`、`dl::Index`、`bool`、`dl::Float16` 或 `dl::BFloat16`。读取不改变原张量的类型、存储或求导历史。CUDA 张量先复制到 CPU，读取操作会等待该数据传输完成。

`data<T>()` 返回原存储地址，`T` 与张量 dtype 对应。它用于底层数据访问，元素位置由张量的 shape、stride 和 offset 决定。CUDA 地址交给设备接口使用。

## 存储与视图

| 操作 | 语义 |
| --- | --- |
| Tensor 复制 | 共享张量身份 |
| transpose / slice / expand | 建立共享存储的布局视图 |
| reshape | 可表示时共享存储，否则按逻辑顺序复制 |
| clone | 建立独立存储 |
| detach | 共享数据并结束求导历史 |
| contiguous | 获得连续布局 |
| to / astype | 显式选择设备或类型 |

shape、stride 和 storage offset 共同决定地址。广播按末尾维对齐；`expand` 以零 stride 表达重复读取。涉及写入的接口遵循各自的别名与拥有权约定。

## 算子

公开声明集中在 [ops.h](../include/dl/ops.h)，包括：

- 算术、比较、选择与常用激活函数。
- sum、mean、amax、softmax、logsumexp 和统计归约。
- matmul、linear、卷积、池化、插值与归一化。
- concat、stack、unbind、repeat、gather、scatter_add 与 Embedding。
- cumsum、稳定排序、topk、常量填充和 one_hot。
- 加权交叉熵、二分类损失、Focal、Dice、KL、MSE、L1、SmoothL1 与 Huber。
- RNN、GRU、LSTM 和缩放点积注意力。

矩阵乘支持向量和批维广播。分类标签采用 int64，交叉熵的 `ignore_index` 不参与 mean 分母。CPU f64 与低精度路径遵循各自的累积类型。

`cumsum(x, axis)` 返回该轴的前缀和，反向计算后缀和。低精度输入累积为 FP32，布尔输入累积为 int64。`sort` 和 `topk` 返回包含 `values` 与 int64 `indices` 的结构；相同值按原索引排序，NaN 升序在末尾、降序在开头。梯度传回被选择的原位置。`topk` 当前通过排序选取前 k 项。

`pad(x, {left, right, top, bottom}, value)` 从最后一维起依次给出两侧填充量，填充量为非负整数。`one_hot(indices, classes)` 在整数索引末尾增加类别轴，默认输出 FP32。数据批次的序列填充见[数据加载](data.md)，损失用法见[损失函数](objectives.md)。

## 一阶求导

```cpp
auto x = dl::ones({4}, {.requires_grad = true});
dl::sum(x * x).backward();
auto dx = *x.grad();
x.zero_grad();
```

`backward` 累计到叶梯度槽。`grad` / `grad_many` 按输入顺序交付显式梯度，适用于组合变换或函数式更新；它们不改写叶梯度槽。非标量输出使用形状与设备匹配的种子。

一次反向消费其可达计算图。保存的张量数据在反向前保持不变，模型迁移和状态加载安排在没有待执行反向图的阶段。共享输入及重复分支的贡献沿统一累计器求和。

梯度 API、范数与裁剪的完整用法见 [C++ 示例](../examples/gradients.cpp)、[C17 显式梯度](../examples/c_gradients.c)和[梯度处理](../examples/c_gradient_processing.c)。
