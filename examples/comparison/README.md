# 跨框架比较示例

这里的程序比较 Ctorch、PyTorch 和 TensorFlow 的训练与计算结果。它们属于教学示例，使用 `CTORCH_BUILD_EXAMPLES=ON` 构建 C++ 入口，Python 框架另行安装。数据、权重、CSV 和 JSON 输出由命令行指定到仓库外。

项目模型定义见 [project_models.h](../project_models.h)。实际结果和实验条件见[对比教程](../../docs/projects/comparison.md)。

## Python 环境

使用 Python 3.12 创建独立环境，安装本次对比使用的版本：

```sh
python3.12 -m venv ../Ctorch-work/frameworks
../Ctorch-work/frameworks/bin/python -m pip install numpy torch==2.11.0 tensorflow==2.20.0
```

Windows 使用 `py -3.12` 创建环境，Python 路径为 `../Ctorch-work/frameworks/Scripts/python.exe`。以下命令中的 `python` 指这个环境中的解释器。

## 比较一批数据

先按[项目教程](../../docs/projects/README.md)准备数据并训练模型，再导出模型权重、输入、输出、参数梯度和一次 SGD 更新结果。例如比较 LeNet：

```sh
../Ctorch-work/build/bin/ctorch_project_compare lenet ../Ctorch-work/runs/mnist-lenet/best.ct ../Ctorch-work/data/mnist ../Ctorch-work/comparison/lenet cpu
python examples/comparison/compare.py ../Ctorch-work/comparison/lenet --framework pytorch
python examples/comparison/compare.py ../Ctorch-work/comparison/lenet --framework tensorflow
```

| 模型名 | 权重 | 数据路径 | 比较输入 |
| --- | --- | --- | --- |
| `mlp` | MNIST MLP 的 `best.ct` | MNIST IDX 目录 | 测试集前 32 张图像 |
| `lenet` | LeNet 的 `best.ct` | MNIST IDX 目录 | 测试集前 16 张图像 |
| `resnet` | ResNet-20 的 `best.ct` | `cifar-10-batches-bin` | 测试集前 4 张图像 |
| `transformer` | 中型字节模型的 `best.ct` | Shakespeare `input.txt` | 测试区间前 2 条长度 128 的序列 |

[project_compare.cpp](project_compare.cpp) 生成 `weights.ct`、`results.ct` 和 `ctorch.json`。[compare.py](compare.py) 读取快照，通过 [frameworks.py](frameworks.py) 在两个 Python 框架中计算同一张网络。每个框架分别生成 JSON，列出所有参数的梯度误差和更新误差。

比较使用 FP32、平均交叉熵与学习率 0.01 的 SGD。所有模型处于评估模式，BatchNorm 使用保存的统计量，dropout 关闭；参数梯度仍然计算。逐元素容差为 `atol=2e-4, rtol=2e-4`，超出时比较程序返回非零退出码。

TensorFlow 的图像计算使用 NHWC 布局，载入时转换卷积权重，导出梯度时转换回 OIHW。步长卷积采用显式对称填充；LeNet 展平前恢复 NCHW 的特征顺序。Transformer 使用相同的因果掩码、LayerNorm、精确 GELU 和共享输出权重。

CPU 比较前设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`。PyTorch 和 TensorFlow 的算子内及算子间线程数在程序中设为 1。依次运行三个框架，以免计时互相竞争。Ctorch 应使用 Release 构建；启用 BLAS 的构建配置见[构建手册](../../docs/build.md)。

计时只包含模型前向，不包含数据读取、格式转换、反向传播或参数更新。测量在一次 SGD 更新后进行，预热 3 次，随后计时 7 组，每组 5 次，取组均值的中位数。CUDA 路径在计时边界同步设备。PyTorch 使用 eager，TensorFlow 使用 eager，均未启用图编译。

在 Ctorch 命令末尾加上 `--reuse-memory`，可以测量开启 `dl::MemoryGuard` 的结果。它复用前向、反向和更新期间释放的缓冲，计算仍使用 eager 算子。例如：

```sh
../Ctorch-work/build/bin/ctorch_project_compare lenet ../Ctorch-work/runs/mnist-lenet/best.ct ../Ctorch-work/data/mnist ../Ctorch-work/comparison/lenet-reuse cpu --reuse-memory
```

`ctorch.json` 用 `reuse_memory` 记录是否启用了此作用域。原理和训练写法见[性能与内存](../../docs/performance.md)。

Ctorch GPU 比较把最后一个参数改为设备编号 `0`，PyTorch 追加 `--device cuda`。本 TensorFlow 示例固定使用 CPU。

## 比较连续训练

下面训练一个 MNIST MLP。Ctorch 导出初始参数和每一步的样本编号，两个 Python 框架读取相同内容。三个程序均训练 200 步，批量 64，使用学习率 0.1、无动量和权重衰减的 SGD。

```sh
../Ctorch-work/build/bin/ctorch_compare_training ../Ctorch-work/data/mnist ../Ctorch-work/comparison/training 200
python examples/comparison/train_mlp.py ../Ctorch-work/data/mnist ../Ctorch-work/comparison/training --framework pytorch
python examples/comparison/train_mlp.py ../Ctorch-work/data/mnist ../Ctorch-work/comparison/training --framework tensorflow
```

使用 CPU 运行。前 55,000 个训练样本参与打乱，种子为 7；模型初始化种子为 42。`order.ct` 保存具体编号，因此 Python 框架无需复现 C++ 随机数发生器。每步交叉熵记录在对应的 `*-curve.csv` 中，完整测试集的损失和准确率记录在 `*-training.json` 中。

C++ 入口为 [train_mlp.cpp](train_mlp.cpp)，Python 入口为 [train_mlp.py](train_mlp.py)。三框架各自进行前向、求导和参数更新，没有从另一个框架复制训练后的参数。
