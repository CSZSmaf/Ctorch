# 张量与梯度

## 类型与设备

CPU 通用浮点计算使用 `f32` / `f64`，CUDA 通用浮点计算使用 `f32`。`f16` / `bf16` 用于低精度存储及选定的矩阵乘、卷积和注意力路径；归约与主参数按算子约定使用 FP32。索引使用 `i64`，掩码使用 `bool`。

通过 `TensorOptions::device` 选择设备，或调用 `Tensor::to(Device)` 显式迁移。一次运算的输入使用相同设备及对应算子要求的 dtype。CUDA 内核布局支持至多 16 维。

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
- 分类交叉熵、二分类损失、MSE、L1、SmoothL1 与 Huber。
- RNN、GRU、LSTM 和缩放点积注意力。

矩阵乘支持向量和批维广播。分类标签采用 int64，交叉熵的 `ignore_index` 不参与 mean 分母。CPU f64 与低精度路径遵循各自的累积类型。

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
