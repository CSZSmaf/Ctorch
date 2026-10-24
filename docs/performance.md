# 性能与内存

模型的耗时包括矩阵计算、张量寻址、数据复制和内存分配。CUDA 后端使用 cuBLAS、cuBLASLt、cuDNN 和通用 CUDA 内核，CPU 后端使用类型化循环和 BLAS。

## CUDA 计算与显存复用

### 合并 Linear 的矩阵乘和偏置

Linear 计算 <span data-tex="Y = XW^{\mathsf T} + b">Y = X Wᵀ + b</span>。如果先计算矩阵乘，再逐元素加偏置，就需要额外提交一次 CUDA 内核，并重新读取和写入输出矩阵。

对连续且地址满足 16 字节对齐的输入、权重和向量偏置，Ctorch 使用一次 cuBLASLt 调用完成这两个操作。它将行主序缓冲视为列主序矩阵，直接计算 <span data-tex="Y^{\mathsf T} = WX^{\mathsf T} + b">Yᵀ = W Xᵀ + b</span>。偏置的列间距设为零，使同一向量用于所有样本。输入、权重和输出都无需为此转置或复制。

同一设备上，相同形状和精度复用已有的矩阵描述和计算方案。首次使用时查询方案，命中缓存的调用只提交数据地址和工作区。缓存最多保存 64 个方案，工作区按所选算法的实际需求分配，上限为 32 MiB，并在训练步之间复用。

这条路径支持 FP32、FP16 和 BF16 输入，低精度乘法使用 FP32 累积和输出。无偏置时直接计算矩阵乘；其他偏置广播形式或特殊布局使用原有矩阵乘和加法。反向传播继续将输入梯度和权重梯度直接写入各自的累计目标，偏置梯度沿样本维求和。

算法与布局的说明见 [cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api)。cuBLASLt 随 CUDA Toolkit 提供，普通 `dl::linear` 和 `dl::nn::Linear` 调用会自动使用适用的实现。

### 让相邻线程读取相邻元素

对形状为 `[rows, columns]` 的连续矩阵，`sum(input, {0})` 沿行求和，得到每一列的总和。如果一组线程只计算一列，相邻线程读取的地址会间隔 `columns` 个元素。列数较多时，一次显存访问中实际用到的数据比例较低。

Ctorch 让相邻线程分别负责相邻列，每个线程先累计 64 行，再用相同的算法合并各分区的结果。这样每轮读取都沿一行连续进行。每个输出位置只有一个线程写入，无需原子加法。`amax` 使用相同的遍历方式。需要分区但并行任务较少时，仍使用原有归约路径，减少内核启动。

连续张量的前导维归约都可以使用这条路径。例如，`[batch, length, width]` 沿前两维求和时，可以视为 `[batch * length, width]` 的矩阵。Linear 偏置梯度和广播梯度中的对应求和也会使用它。转置、切片和其他归约轴继续按张量步长访问。

分区大小是编译期常量，求和与最大值通过模板选择，内层循环只执行所需运算。每个线程用四个独立累加器处理不同的行，最后两两合并，减少连续依赖和连续舍入。分区会改变浮点加法的顺序，结果可能有少量舍入差异。

在 NVIDIA GeForce RTX 5090 Laptop GPU 上，对连续的 `4096 × 1024` FP32 张量测得以下 CUDA event 耗时。两个版本都启用相同的缓冲复用。每个算子预热至少 200 ms，再测量 7 组，每组执行 20 次；表中取三轮独立测量的中位数。

| 运算 | 原实现 / μs | 当前实现 / μs | 加速比 |
| --- | ---: | ---: | ---: |
| 跨行求和 `sum(input, {0})` | 82.04 | 18.48 | 4.44× |
| 跨行最大值 `amax(input, {0})` | 81.43 | 19.38 | 4.20× |

### 在训练步之间保留空闲显存

CUDA 张量默认使用专用的异步内存池。张量释放后，空闲空间留在池中供后续分配复用。读取损失或等待 GPU 完成时，内存池也继续保留这些空间。普通 CUDA 训练无需额外添加内存作用域。

`dl::cuda::memory_stats()` 的 `used_bytes` 表示仍在使用的分配，`reserved_bytes` 还包含池内的空闲空间。一个训练阶段结束后，可以调用 `dl::cuda::release_unused_memory()` 释放 cuBLASLt 和 cuDNN 的缓存工作区，并归还池内空闲空间。仍被张量持有的存储继续有效。

```cpp
dl::synchronize(dl::Device::cuda());
const auto stats = dl::cuda::memory_stats();
std::cout << stats.used_bytes << ' ' << stats.reserved_bytes << '\n';
dl::cuda::release_unused_memory();
```

需要限制缓存保留量时，可以调用 `dl::cuda::set_memory_release_threshold(bytes)`。设为零后，Runtime 会在同步点尝试回收空闲空间。阈值控制缓存保留策略，不是整个程序的显存容量上限。完整示例见 [cuda_memory.cpp](../examples/cuda_memory.cpp)。

### 转交反向传播中的梯度缓冲

`reshape`、`flatten`、`squeeze` 和 `unsqueeze` 的反向传播只需恢复原输入形状，梯度的元素值不变。`clone` 的反向也直接传递梯度。如果每经过一个这样的节点都复制梯度，就会增加临时显存和设备复制操作。

Ctorch 在这些节点之间转交独立且连续的梯度缓冲。C++ 的移动操作转交存储句柄，显存中的元素保持在原位置。首次贡献直接成为目标梯度，已有贡献则继续原地累计。类型转换生成的独立结果和默认标量种子也使用同一套转交方式。

通过 `dl::grad` 请求的中间梯度需要保留其值。存储仍由其他结果持有时，累计目标使用独立副本。调用方传入的反向种子也保持独立。这样，求中间梯度、分支汇合和多次梯度累加仍使用原来的接口和数值规则。

### 批量更新参数

CUDA 后端将最多 32 个连续参数的更新描述放入同一份内核参数表。一个参数保留自己的学习率、权重衰减和状态，多个参数可以在同一次提交中更新。SGD、Adam、AdamW、RMSprop 和 Adagrad 共用这套遍历方式。

批量内核中，每个线程一次处理四个元素，并复用参数描述和更新系数。各轮中相邻线程访问相邻元素，循环次数在编译期展开。每个元素继续使用原来的动量、矩估计和参数更新公式。张量末尾按实际元素数处理，带步长的参数使用对应的布局内核。

线程内处理多个元素的方式可参考 [NVIDIA 的执行配置说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations)和 [PyTorch 的多张量更新实现](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh)。

### 测量完整训练步

单个算子变快后，完整训练的收益还取决于它原本占用多少时间。矩阵乘、卷积、数据传输和优化器更新仍然需要执行。测量时将输入提前放到 GPU，先预热，再对包含清空梯度、前向、反向和参数更新的整个训练步计时。

CUDA 算子异步提交。宿主计时需要在测量边界等待 GPU 完成；CUDA event 则应记录在执行计算的 stream 上。调用 `loss.item<float>()` 会把标量传回 CPU，适合按日志间隔使用。

MLP、卷积网络和单层编码器的完整 CUDA 训练步耗时见[框架结果对比](projects/comparison.md#cuda-训练步耗时)。

内存池的行为见 [CUDA 异步内存分配说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)。

## 在 CPU 训练循环中复用内存

在循环外创建一个 `dl::MemoryGuard`。作用域内的普通张量分配会使用同一个缓冲池，前向、反向和优化器更新都可以复用空闲块。

下面是完整的 CPU 训练程序：

```cpp
#include <dl/dl.h>
#include <iostream>

int main() {
    dl::manual_seed(42);
    auto input = dl::randn({64, 3});
    auto target = dl::sum(input, {1}, dl::KeepDim::yes);
    dl::nn::Linear model(3, 1);
    dl::optim::SGD optimizer(model.parameters(), {.learning_rate = 0.05});

    dl::MemoryGuard memory;
    for (int step = 0; step < 100; ++step) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(model(input), target);
        loss.backward();
        optimizer.step();
        if (step % 20 == 0) std::cout << loss.item<float>() << '\n';
    }

    const auto stats = memory.memory_stats();
    std::cout << "reserved bytes: " << stats.reserved_bytes << '\n';
    std::cout << "in-use bytes: " << stats.in_use_bytes << '\n';
    memory.release_unused_memory();
}
```

构建方式见[第一次训练示例](../examples/first_training.cpp)和[构建手册](build.md)。对已有程序，通常只需在反复执行的计算外增加 `dl::MemoryGuard memory;`。

作用域应覆盖整个循环。若每次迭代都重新创建作用域，空闲块也会在每次迭代结束时释放。

## 缓冲何时可以复用

张量、视图和反向传播保存值共同持有存储。最后一个持有者释放后，这块存储才进入空闲状态。例如，前向结果仍被损失函数保存时，下一次计算会使用其他缓冲。

`MemoryGuard` 退出时释放空闲块。仍被张量持有的块会继续存活，因此输出可以返回到作用域外。每个执行线程创建自己的作用域，张量的最后一个引用可以在消费线程释放。嵌套作用域结束后恢复外层缓冲池。

`memory_stats()` 的 `reserved_bytes` 包含空闲和使用中的存储，`in_use_bytes` 只包含仍被持有的部分。输入尺寸经常变化时，池中会积累多种大小的块，可以在一个计算阶段结束后调用 `release_unused_memory()` 归还空闲块。

固定执行计划已有自己的工作区和 `with_memory` 方法。它与普通训练作用域使用同一套缓冲管理机制，见[组合与执行](execution.md)。

## C++ 循环如何影响性能

### 将布局判断放在内层循环之外

连续张量的地址逐元素递增。广播标量的地址保持不变。把这两种步长在循环外确定下来，内层就只需读取、计算和写入。Ctorch 用局部模板区分步长为 0、1 和一般步长的情况，使常见路径的步长成为编译期常量。

ReLU 等简单操作使用具有具体类型的 lambda。编译器能够看到操作体，更容易将它内联到数据循环中。模板本身不会在每个元素上执行额外的运行时分派。

### 减少 Transformer 中重复的逐元素计算

Dropout 先生成布尔掩码，再根据掩码保留并缩放输入。数据指针在进入循环前取得，前向和反向共用已有的二元遍历函数。连续输入走固定步长路径，转置和切片沿各自的步长访问。

因果注意力通常用负无穷屏蔽每行末尾的未来位置。Softmax 可以直接将这些位置写为零，只对前面的有效范围计算指数和归一化。LogSoftmax 对应的位置写为负无穷。整行都为负无穷时，结果为 NaN。

GELU 使用误差函数 `std::erf`。公式中的固定系数用 `constexpr` 表示，循环通过乘法计算缩放后的输入和高斯密度。改写除法会产生小幅浮点舍入差异，因此比较时需要同时检查前向、梯度和连续更新后的参数。

### 按行处理卷积和池化

卷积展开窗口时，同一行共用输入起点和有效列范围。连续部分使用 `std::copy_n`，带步长的视图则沿固定步长访问。池化按平面、行、列遍历，减少从线性下标反复解算空间坐标的整数除法。

归约到同一个输出的行使用局部累加器。内层计算完成后写回结果，保持原有求和顺序。

### 区分表达式与实际计算次数

下面的表达式很短，但在 eager 张量接口中会执行多次算子调用：

```cpp
auto output = ((input - mean) * inverse_std) * weight + bias;
```

每个算子通常产生一个结果张量。C++ 编译器看到的是这些函数调用，不能自动把运行时的张量计算图合并成一个循环。缓冲复用减少分配和释放；算子融合则进一步减少中间结果的读写。两者解决的开销不同。

Ctorch 保留独立输出与原地更新的接口区别。原地累计可以读写同一份存储，因此通用循环使用普通指针，不给这些路径附加不重叠假设。

## 比较实际模型

先固定输入形状、精度、线程数、后端和模型模式，再比较完整模型。推理耗时与训练步耗时分别记录，文件读取和权重加载放在计时范围之外。

[框架结果对比](projects/comparison.md#cpu-前向耗时)同时列出普通分配、启用缓冲复用、PyTorch 和 TensorFlow 的结果。[比较程序](../examples/comparison/README.md)提供复现命令。缓冲复用效果取决于模型的临时张量数量、大小和系统分配器。

相关实现方式可参考 [PyTorch 的 CPU 逐元素循环](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Loops.h)和 [Eigen 的表达式求值与临时结果](https://libeigen.gitlab.io/eigen/docs-nightly/TopicLazyEvaluation.html)。

GELU 的实现还可参考 [PyTorch CPU 激活函数](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Activation.cpp)。
