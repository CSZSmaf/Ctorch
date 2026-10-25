# 项目教程

这组项目使用公开数据训练模型，包含数据读取、训练、验证、保存、恢复和预测。先完成基础教程中的[训练](../tutorial/index.html#training)与[保存恢复](../tutorial/index.html#state)，再选择一个项目运行。

| 项目 | 数据与任务 | 参数量 | 主要内容 |
| --- | --- | ---: | --- |
| [MNIST MLP](mnist.md#mlp) | 手写数字分类 | 101,770 | 完整分类流程、训练指标、三框架训练对比 |
| [MNIST LeNet](mnist.md#lenet) | 手写数字分类 | 44,426 | 卷积、池化、特征尺寸计算 |
| [CIFAR-10 ResNet-20](resnet.md) | 彩色图像分类 | 272,474 | 残差模块、BatchNorm、分阶段降采样 |
| [Shakespeare Transformer](transformer.md) | 下一个字节预测与文本生成 | 842,496 | 因果注意力、共享词表、梯度裁剪、自回归生成 |

MLP 和 LeNet 适合第一次运行真实数据训练。ResNet 和 Transformer 使用更多模块，适合练习如何组织一个完整的单机训练项目。[框架结果对比](comparison.md)提供 Ctorch、PyTorch 和 TensorFlow 的实际训练结果、数值误差与 CPU 推理耗时。

[多维性能比较](performance.md)进一步比较不同规模、精度与执行配置下的 CUDA 和 CPU 推理、完整训练、显存及首次调用开销。[归一化与完整训练](normalization.md)比较 BatchNorm 通道归约、直接写入梯度和 PyTorch 编译配置，并说明编码器的数值差异。

## 构建程序

在仓库根目录运行。以下路径把构建、数据和输出统一放在仓库外的 `../Ctorch-work` 中。

```sh
cmake -S . -B ../Ctorch-work/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCTORCH_BUILD_EXAMPLES=ON
cmake --build ../Ctorch-work/build
```

程序位于 `../Ctorch-work/build/bin`。Windows 下可执行文件带有 `.exe` 后缀。CPU 矩阵乘法可以通过 `-DCTORCH_USE_BLAS=ON` 使用已安装的 BLAS 库。各项目默认使用 CPU。

使用 NVIDIA GPU 时，在另一个构建目录启用 CUDA 和 cuDNN：

```sh
cmake -S . -B ../Ctorch-work/build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DCTORCH_BUILD_EXAMPLES=ON -DCTORCH_USE_CUDA=ON -DCTORCH_USE_CUDNN=ON
cmake --build ../Ctorch-work/build-gpu
```

CUDA、cuDNN 的依赖和路径设置见[构建手册](../build.md)。随后把项目命令中的 `build/bin` 改为 `build-gpu/bin`，并追加 `--device 0`。项目中的实测训练使用 FP32；`--precision bf16` 或 `--precision f16` 可以另行启用混合精度。

## 获取数据

[下载脚本](../../examples/download_data.py)使用 Python 3.12 或更高版本的标准库，不需要安装 Python 深度学习框架。每条命令只下载指定数据集。

```sh
python examples/download_data.py mnist ../Ctorch-work/data/mnist
python examples/download_data.py cifar10 ../Ctorch-work/data/cifar10
python examples/download_data.py shakespeare ../Ctorch-work/data/shakespeare
```

CIFAR-10 使用 binary 版本，解压目录为 `data/cifar10/cifar-10-batches-bin`。Shakespeare 文本保存在 `data/shakespeare/input.txt`。已有数据可以直接传给训练程序。

## 阅读输出

所有训练程序都接受一个输出目录。训练结束后主要包含以下文件：

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `initial.ct` | 初始参数和缓冲区 | 比较训练前后的模型 |
| `last.ct` | 最后一次更新的模型 | 检查最终参数 |
| `best.ct` | 验证损失最低的模型 | 测试或预测 |
| `latest.train.ct` | 参数、优化器、采样器、随机状态和训练进度 | 继续训练 |
| `metrics.csv` | 各评估点的训练和验证指标 | 观察收敛过程 |
| `summary.json` | 配置、更新次数及最终测试结果 | 记录一次运行 |

程序按验证损失选择 `best.ct`，训练完成后加载它评估测试集。分类准确率按样本数加权，文本指标按有效目标字节数加权。

`--stop-after N` 在第 N 次参数更新后保存训练状态。下一次运行使用 `--resume <latest.train.ct>` 继续。保持模型结构、数据、批量大小和精度一致；`--epochs` 或 `--steps` 表示恢复后的总训练目标。

## 代码组织

模型集中在 [project_models.h](../../examples/project_models.h)。每个可执行程序只选择数据和模型，通用逻辑由以下文件提供：

| 文件 | 负责的内容 |
| --- | --- |
| [classification_training.h](../../examples/classification_training.h) | 图像分类的训练、验证和测试 |
| [training.h](../../examples/training.h) | 微批次、精度、指标、检查点和最佳模型选择 |
| [mnist_training.h](../../examples/mnist_training.h) | MNIST 数据划分 |
| [cifar_data.h](../../examples/cifar_data.h) | CIFAR 二进制读取与归一化 |
| [text_training.h](../../examples/text_training.h) | 文本划分和语言模型训练 |
| [text_data.h](../../examples/text_data.h) | 字节序列和下一字节标签 |

跨框架比较程序单独位于 [examples/comparison](../../examples/comparison/README.md)。它们使用项目中的同一组模型，不参与 Ctorch 库的构建依赖。
