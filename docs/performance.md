# 性能与内存

模型的耗时包括矩阵计算、张量寻址、数据复制和内存分配。Ctorch 的 CPU 内核使用类型化循环和 BLAS。训练代码还可以通过 `MemoryGuard` 复用中间张量的存储。

## 在训练循环中复用内存

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
