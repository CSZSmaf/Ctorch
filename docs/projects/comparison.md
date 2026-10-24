# Ctorch、PyTorch 与 TensorFlow 的结果对比

本页比较三个框架训练 MNIST MLP 的结果，以及四种模型在相同权重和输入下的计算结果。

## 相同初值和批次的训练结果

模型为 `784 → 128 → 10` 的 MLP，隐藏层使用 ReLU。三个框架读取同一份初始参数和每一步的样本编号，各自进行前向、反向和 SGD 更新。

| 设置 | 数值 |
| --- | --- |
| 训练样本来源 | MNIST 训练集前 55,000 张，打乱种子 7 |
| 模型初始化 | Ctorch 种子 42 初始化后导出，三个框架共用 |
| 输入 | FP32，像素除以 255 |
| 批量与更新次数 | 每批 64，连续 200 步 |
| 优化器 | SGD，学习率 0.1，无动量和权重衰减 |
| 损失 | 十分类平均交叉熵 |
| 最终评估 | 官方测试集全部 10,000 张图像 |

| 框架 | 测试交叉熵 | 测试准确率 | 相对 Ctorch 的最大参数差 |
| --- | ---: | ---: | ---: |
| Ctorch | 0.41271493 | 89.01% | — |
| PyTorch 2.11.0 | 0.41271558 | 89.01% | 0.00015813 |
| TensorFlow 2.20.0 | 0.41271558 | 89.01% | 0.00015813 |

三组训练得到相同的测试准确率，损失接近，参数仍有浮点差异。本实验训练 200 步；[MNIST 项目](mnist.md#实际训练结果)中的 97.05% 是另一次使用 AdamW 训练 5 轮的结果。

## 逐步查看训练损失

曲线记录每次更新前当前批次的交叉熵。不同批次的难度不同，因此曲线会波动。三条曲线非常接近，可以切换框架或拖动步数查看数值。

<div class="lab">
<div class="comparison-controls"><label for="comparison-framework">显示框架</label><select id="comparison-framework"><option value="all">全部框架</option><option value="ctorch">Ctorch</option><option value="pytorch">PyTorch</option><option value="tensorflow">TensorFlow</option></select></div>
<div class="comparison-key"><span style="--curve-color:var(--accent)">Ctorch</span><span style="--curve-color:var(--green)">PyTorch</span><span style="--curve-color:var(--blue)">TensorFlow</span></div>
<svg id="comparison-chart" class="comparison-chart" viewBox="0 0 760 270" role="img" aria-label="三个框架的 MLP 训练损失曲线"></svg>
<label class="range-label" for="comparison-step">更新步数 <output id="comparison-step-value" for="comparison-step">1</output></label>
<input id="comparison-step" type="range" min="1" max="200" value="1">
<table><thead><tr><th>框架</th><th>当前批次交叉熵</th></tr></thead><tbody aria-live="polite"><tr><td>Ctorch</td><td id="comparison-loss-ctorch">2.298443</td></tr><tr><td>PyTorch</td><td id="comparison-loss-pytorch">2.298443</td></tr><tr><td>TensorFlow</td><td id="comparison-loss-tensorflow">2.298443</td></tr></tbody></table>
</div>

## 四种模型的数值比较

这组实验分别读取四个项目训练得到的权重。三个框架使用相同输入、权重、标签、FP32 精度和平均交叉熵，比较 logits、所有可训练参数的梯度，以及学习率为 0.01 的一次 SGD 更新。

BatchNorm 使用保存的均值和方差，dropout 关闭。梯度仍然计算。这个设置让模型的计算不受批次统计量更新或随机 dropout 掩码影响。

| 模型 | 输入批量 | 数据 |
| --- | ---: | --- |
| MLP | 32 | MNIST 测试集前 32 张 |
| LeNet | 16 | MNIST 测试集前 16 张 |
| ResNet-20 | 4 | CIFAR-10 测试集前 4 张 |
| Transformer | 2×128 | 文本测试区间前两条序列 |

下表分别列出相对 Ctorch 的最大绝对误差。梯度和更新列取该模型全部参数中的最大值。

| 模型 | 对比框架 | logits 误差 | 交叉熵误差 | 参数梯度误差 | SGD 后参数误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | PyTorch | 5.72e-6 | 1.49e-8 | 5.59e-8 | 7.45e-9 |
| MLP | TensorFlow | 5.72e-6 | 4.47e-8 | 5.77e-8 | 1.49e-8 |
| LeNet | PyTorch | 3.81e-6 | 2.25e-8 | 1.17e-7 | 1.49e-8 |
| LeNet | TensorFlow | 4.77e-6 | 1.16e-10 | 2.46e-7 | 1.49e-8 |
| ResNet-20 | PyTorch | 1.91e-6 | 8.94e-8 | 1.31e-6 | 5.96e-8 |
| ResNet-20 | TensorFlow | 2.38e-6 | 4.47e-8 | 1.13e-6 | 2.98e-8 |
| Transformer | PyTorch | 8.58e-6 | 4.77e-7 | 3.54e-7 | 1.19e-7 |
| Transformer | TensorFlow | 8.11e-6 | 3.58e-7 | 3.35e-7 | 2.98e-8 |

所有张量都通过 `atol=2e-4, rtol=2e-4` 的逐元素比较。公式为：

<span data-tex="|a-b|\le 2\times10^{-4}+2\times10^{-4}|b|">绝对差 ≤ 2e-4 + 2e-4 × 参考值绝对值</span>

这些误差说明上述输入上的前向、反向和一次参数更新接近。完整训练结果在前一节单独比较。

## CPU 前向耗时

在 Intel Core Ultra 9 275HX、Windows 11 上依次运行三个框架。CPU 算子内与算子间线程数设为 1，输入已在内存中。Ctorch 使用 GCC 16.1 Release 与 OpenBLAS 0.3.34；PyTorch 使用 2.11.0，TensorFlow 使用 2.20.0。

测量使用上节的批量和模型，在一次 SGD 更新后计时。先预热 3 次，再运行 7 组、每组 5 次前向，取每组平均耗时的中位数。计时包含模型调用，不包含读取文件、载入权重、初始布局转换、反向传播或参数更新。

| 模型 | Ctorch / ms | PyTorch / ms | TensorFlow / ms |
| --- | ---: | ---: | ---: |
| MLP，批量 32 | 0.068 | 0.059 | 0.232 |
| LeNet，批量 16 | 1.144 | 0.548 | 0.732 |
| ResNet-20，批量 4 | 14.634 | 3.886 | 8.988 |
| Transformer，批量 2、长度 128 | 13.561 | 5.669 | 13.584 |

在这个配置下，Ctorch 的 MLP 前向耗时接近 PyTorch，Transformer 接近 TensorFlow。卷积模型与 PyTorch 仍有明显差距。批量、布局、设备和计算后端都会影响耗时。

PyTorch 和 TensorFlow 均直接使用 eager 算子，没有开启 `torch.compile` 或 `tf.function`。PyTorch 的 Python 环境为 3.14，TensorFlow 为 3.12。Ctorch/PyTorch 的图像布局是 NCHW，TensorFlow 使用 NHWC，布局转换在比较程序载入数据时完成。三个 Transformer 都使用普通矩阵运算实现因果注意力。

三框架的这张表统一使用 CPU。TensorFlow 2.20 的原生 Windows 包不提供 CUDA GPU 支持；运行其 GPU 版本可以按[官方安装说明](https://www.tensorflow.org/install/pip)使用 WSL2 或 Linux。

## 复现比较

[比较示例说明](../../examples/comparison/README.md)列出环境安装、四种模型的数据路径和全部运行命令。源码分别负责以下任务：

| 程序 | 作用 |
| --- | --- |
| [project_compare.cpp](../../examples/comparison/project_compare.cpp) | Ctorch 前向、求导、SGD 更新、计时和结果导出 |
| [frameworks.py](../../examples/comparison/frameworks.py) | 两个 Python 框架的模型计算 |
| [compare.py](../../examples/comparison/compare.py) | 读取相同输入和权重，比较所有结果张量 |
| [train_mlp.cpp](../../examples/comparison/train_mlp.cpp) | 导出初值和采样顺序，执行 Ctorch 连续训练 |
| [train_mlp.py](../../examples/comparison/train_mlp.py) | 使用相同初值和批次执行 Python 框架训练 |

所有程序把权重、曲线和 JSON 写到指定输出目录。比较示例不加入库的运行依赖，训练所需的数据也不保存在源码仓库中。
