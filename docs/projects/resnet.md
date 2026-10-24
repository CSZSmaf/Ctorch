# CIFAR-10：ResNet-20

本项目训练一个彩色图像分类器。相比 MNIST，它增加了残差连接、BatchNorm 和多阶段特征提取。输入为 32×32 RGB 图像，输出为十个类别分数。

## 数据准备

[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) 包含 50,000 张训练图像和 10,000 张测试图像。下载 binary 版本后，五个训练文件各包含 10,000 条记录。每条记录由一个标签字节和 3,072 个像素字节组成，像素依次存放红、绿、蓝三个通道。

```sh
python examples/download_data.py cifar10 ../Ctorch-work/data/cifar10
```

[cifar_data.h](../../examples/cifar_data.h) 将像素读取为 `[N,3,32,32]`，并使用 `pixel / 127.5 - 1` 缩放到 `[-1,1]`。前 45,000 张训练图像用于更新，后 5,000 张用于验证。测试集单独用于最终评估。

本次训练使用原始图像，不做随机裁剪或翻转。增加增强时，只对训练样本应用随机变换，验证和预测继续使用固定的归一化。

## 网络结构

网络有三个残差阶段，每个阶段包含三个双卷积块。两个阶段交界处使用步长 2 将空间尺寸减半，通道数增加一倍。

| 部分 | 输出形状 |
| --- | --- |
| 3×3 卷积 + BatchNorm + ReLU | `[N,16,32,32]` |
| 3 个残差块，16 通道 | `[N,16,32,32]` |
| 3 个残差块，32 通道 | `[N,32,16,16]` |
| 3 个残差块，64 通道 | `[N,64,8,8]` |
| 全局平均池化 + Flatten | `[N,64]` |
| Linear(64,10) | `[N,10]` |

主路径包含 19 个卷积层和一个全连接层。降采样块的捷径使用 1×1 卷积和 BatchNorm 投影，网络共 272,474 个可训练参数。原始 CIFAR ResNet 也讨论了使用补零捷径的方案；这里选择投影捷径。

## 编写残差块

一个残差块先对输入做两次卷积，再与捷径相加：

<span data-tex="y=\operatorname{ReLU}(F(x)+S(x))">y = ReLU(F(x) + S(x))</span>

当形状不变时，`S(x)` 直接返回输入。降采样时，`S(x)` 使用步长 2 的 1×1 卷积，使两条路径形状一致。完整定义在 [project_models.h](../../examples/project_models.h) 的 `ResidualBlock` 中。

```cpp
dl::Tensor forward(const dl::Tensor& input) override {
    auto residual = shortcut_ ? (*shortcut_)(input) : input;
    auto hidden = dl::relu((*first_norm_)((*first_)(input)));
    return dl::relu((*second_norm_)((*second_)(hidden)) + residual);
}
```

BatchNorm 在训练时使用当前批次统计量，并更新均值和方差缓冲区。验证时切换到 `eval()`，使用保存的统计量。保存模型时，参数和这些缓冲区一起写入 `best.ct`。

## 训练与评估

完成[项目构建](README.md#构建程序)后运行：

```sh
../Ctorch-work/build/bin/ctorch_cifar_resnet ../Ctorch-work/data/cifar10/cifar-10-batches-bin ../Ctorch-work/runs/cifar-resnet --epochs 5 --batch-size 128
```

GPU 构建使用 `build-gpu/bin` 中的程序并追加 `--device 0`。示例采用 AdamW，学习率 0.001，权重衰减 0.01，FP32 计算。每轮验证一次，按最低验证交叉熵选择最终测试模型。

这次 FP32 运行训练 5 轮，共 1,760 次更新。测试使用官方 10,000 张图像，结果如下：

| 指标 | 结果 |
| --- | ---: |
| 初始验证交叉熵 | 2.30639 |
| 最佳验证交叉熵 | 0.72379 |
| 测试交叉熵 | 0.76356 |
| 测试准确率 | 73.29% |

这组结果对应上述 5 轮、无数据增强的配置。更长训练与原始 ResNet 论文的训练方案有不同的计算预算，比较准确率时应同时列出训练设置。

## 读取预测

标签 0 到 9 依次为飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。下面展示如何对测试集第一张图像进行预测：

```cpp
auto model = examples::resnet20();
model->load_state_dict(dl::io::load("../Ctorch-work/runs/cifar-resnet/best.ct"));
auto test = examples::load_cifar10(
    "../Ctorch-work/data/cifar10/cifar-10-batches-bin", false);
model->eval();
dl::NoGradGuard no_grad;
auto logits = (*model)(test.input.narrow(0, 0, 1));
auto category = dl::argmax(logits, 1).item<dl::Index>();
```

这段代码需要包含 `project_models.h` 和 `cifar_data.h`。为自己的 RGB 图像分类时，将它缩放到 32×32，按通道排列，并应用相同的 `pixel / 127.5 - 1` 预处理。

## 扩展训练配置

更长训练可以增加 `--epochs`。中断恢复使用 `--resume <latest.train.ct>`。调整训练预算时，先观察 `metrics.csv` 中的训练与验证损失，再决定是否加入数据增强或学习率衰减。

微批次会改变 BatchNorm 每次看到的样本数量，因此 `--micro-batch-size` 不只影响显存占用，也可能改变训练结果。比较微批次配置时需要重新记录验证指标。

代码入口为 [cifar_resnet.cpp](../../examples/cifar_resnet.cpp)。分类训练流程参考 [PyTorch CIFAR-10 教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)和 [TensorFlow CNN 教程](https://www.tensorflow.org/tutorials/images/cnn)。残差网络结构参考 [原始 ResNet 项目](https://github.com/KaimingHe/deep-residual-networks)。
