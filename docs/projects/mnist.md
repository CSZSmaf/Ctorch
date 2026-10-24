# MNIST：MLP 与 LeNet

本项目训练一个手写数字分类器。输入是一张 28×28 的灰度图，输出是数字 0 到 9 的十个分数。先运行 MLP，再用 LeNet 风格卷积网络比较结构和训练结果。

## 数据与标签

MNIST 包含 60,000 张训练图像和 10,000 张测试图像。示例把前 55,000 张训练图像用于更新，后 5,000 张用于验证。像素除以 255 后变为 `[0,1]` 范围内的 FP32 数值。标签保留为整数。

一个批次的输入形状是 `[N,1,28,28]`，标签形状是 `[N]`。模型返回 `[N,10]` 的 logits。交叉熵直接接收 logits，不需要先执行 softmax。

```cpp
auto logits = (*model)(batch.input);
auto loss = dl::cross_entropy(logits, batch.target);
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

## MLP

MLP 将图像展平成 784 个数，用 128 个隐藏单元生成十个类别分数。它的参数量为 101,770。

| 层 | 输出形状 |
| --- | --- |
| Flatten | `[N,784]` |
| Linear(784,128) + ReLU | `[N,128]` |
| Linear(128,10) | `[N,10]` |

完整入口是 [mnist_mlp.cpp](../../examples/mnist_mlp.cpp)，模型定义在 [project_models.h](../../examples/project_models.h) 的 `mnist_mlp()` 中。

按[项目准备](README.md)完成构建与下载后运行：

```sh
../Ctorch-work/build/bin/ctorch_mnist_mlp ../Ctorch-work/data/mnist ../Ctorch-work/runs/mnist-mlp --epochs 5 --batch-size 128
```

模型初始化使用种子 42，训练样本打乱使用种子 7。优化器是 AdamW，学习率为 0.001，权重衰减为 0.01。每轮结束后验证一次。

## LeNet

LeNet 使用局部卷积提取笔画特征，然后进行分类。本示例采用 28×28 输入、ReLU 和最大池化，是 LeNet 风格的现代实现。参数量为 44,426。

| 层 | 输出形状 |
| --- | --- |
| Conv2d(1,6,5) + ReLU | `[N,6,24,24]` |
| MaxPool2d(2) | `[N,6,12,12]` |
| Conv2d(6,16,5) + ReLU | `[N,16,8,8]` |
| MaxPool2d(2) | `[N,16,4,4]` |
| Flatten | `[N,256]` |
| Linear(256,120) + ReLU | `[N,120]` |
| Linear(120,84) + ReLU | `[N,84]` |
| Linear(84,10) | `[N,10]` |

卷积不填充边界，5×5 卷积会使宽和高各减少 4。第二次池化后共有 `16×4×4=256` 个特征，因此第一个全连接层接收 256 维输入。

入口 [mnist_lenet.cpp](../../examples/mnist_lenet.cpp) 与 MLP 共用数据划分和训练流程：

```sh
../Ctorch-work/build/bin/ctorch_mnist_lenet ../Ctorch-work/data/mnist ../Ctorch-work/runs/mnist-lenet --epochs 3 --batch-size 128
```

## 实际训练结果

下表来自完整 MNIST 测试集的 10,000 张图像。两次训练都使用 FP32、批量 128、AdamW 学习率 0.001，在 CUDA/cuDNN 后端运行。评估模型由验证损失选出。

| 模型 | 训练轮数 | 更新次数 | 测试交叉熵 | 测试准确率 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 5 | 2,150 | 0.10409 | 97.05% |
| LeNet | 3 | 1,290 | 0.06327 | 98.05% |

这是两组具体训练配置的结果。比较网络结构时，可以把轮数、优化器和随机种子设为一致，并重复运行。三个框架使用完全相同初值和批次进行 200 步训练的结果见[框架对比](comparison.md#相同初值和批次的训练结果)。

## 预测一张图像

下面读取官方测试集中的第一张图像。创建的模型必须与权重对应；加载 MLP 时使用 `examples::mnist_mlp()`，加载 LeNet 时使用 `examples::lenet()`。

```cpp
#include "project_models.h"
#include <iostream>

int main() {
    auto model = examples::lenet();
    model->load_state_dict(dl::io::load("../Ctorch-work/runs/mnist-lenet/best.ct"));
    auto test = dl::data::load_mnist(
        "../Ctorch-work/data/mnist/t10k-images-idx3-ubyte",
        "../Ctorch-work/data/mnist/t10k-labels-idx1-ubyte");
    model->eval();
    dl::NoGradGuard no_grad;
    auto logits = (*model)(test.input.narrow(0, 0, 1));
    auto prediction = dl::argmax(logits, 1);
    std::cout << prediction.item<dl::Index>() << '\n';
}
```

将自己的图像转成 `[1,1,28,28]` 的灰度张量时，保持与 MNIST 相同的亮度方向和 `[0,1]` 缩放。MNIST 的数字笔画较亮、背景较暗。图像尺寸和预处理发生变化时，模型看到的输入分布也会变化。

## 暂停与恢复

以下命令在 500 次更新后保存状态，再继续到第 5 轮。两次运行使用同一个输出目录。

```sh
../Ctorch-work/build/bin/ctorch_mnist_mlp ../Ctorch-work/data/mnist ../Ctorch-work/runs/mnist-resume --epochs 5 --batch-size 128 --stop-after 500
../Ctorch-work/build/bin/ctorch_mnist_mlp ../Ctorch-work/data/mnist ../Ctorch-work/runs/mnist-resume --epochs 5 --batch-size 128 --resume ../Ctorch-work/runs/mnist-resume/latest.train.ct
```

`best.ct` 用于预测；`latest.train.ct` 用于恢复 AdamW 的动量、采样位置和训练进度。输出文件的完整说明见[项目目录](README.md#阅读输出)。

## 可以继续尝试的修改

把 MLP 隐藏宽度改为 64 或 256，观察参数量和验证损失。对 LeNet，可以比较最大池化和平均池化。修改模型后使用新的输出目录，按相同的数据划分比较结果。

分类流程可参考 [TensorFlow 图像分类教程](https://www.tensorflow.org/tutorials/keras/classification)。本项目的网络、训练命令和上表指标来自 Ctorch 实现。
