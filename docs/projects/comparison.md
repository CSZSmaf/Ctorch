# Ctorch、PyTorch 与 TensorFlow 的结果对比

本页比较三个框架训练 MNIST MLP 的结果，以及四种模型在相同权重和输入下的计算结果，并列出 CPU 与 CUDA 耗时。

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
| Transformer | PyTorch | 7.87e-6 | 0 | 3.69e-7 | 1.19e-7 |
| Transformer | TensorFlow | 8.58e-6 | 1.19e-7 | 3.50e-7 | 2.98e-8 |

所有张量都通过 `atol=2e-4, rtol=2e-4` 的逐元素比较。公式为：

<span data-tex="|a-b|\le 2\times10^{-4}+2\times10^{-4}|b|">绝对差 ≤ 2e-4 + 2e-4 × 参考值绝对值</span>

前一节比较 MNIST 的完整训练结果，后面还列出了 Transformer 的连续参数更新结果。

## CUDA 训练步耗时

在 NVIDIA GeForce RTX 5090 Laptop GPU、Windows 11 上比较四个固定输入的训练循环。Ctorch 使用 MSVC 19.50 Release、CUDA 13.2 编译工具、CUDA 13.4 运行库和 cuDNN 9.21；PyTorch 使用 2.11.0+cu128。两个框架都使用 FP32，矩阵计算关闭 TF32，输入和参数提前放到 GPU。

| 模型 | 输入与结构 | 优化器 |
| --- | --- | --- |
| MLP | 批量 256；`1024 → 2048 → 1000`，隐藏层使用 ReLU | SGD，学习率 0.001，动量 0.9 |
| 卷积加大分类头 | 批量 32；输入 `3 × 64 × 64`；32 通道的 3×3 卷积、ReLU、2×2 平均池化、1000 类全连接层 | AdamW，学习率 0.001，权重衰减 0.01 |
| 三层卷积加全局平均池化 | 批量 32；输入 `3 × 64 × 64`；卷积通道 `3 → 32 → 64 → 64`，每层后使用 ReLU；第一层后进行 2×2 平均池化，最后进行全局平均池化和 1000 类分类 | AdamW，学习率 0.001，权重衰减 0.01 |
| 单层编码器 | 批量 8，长度 128，宽度 256，8 个注意力头，前馈宽度 1024，输出 512 类 | AdamW，学习率 0.001，权重衰减 0.01 |

两种卷积网络均使用 3×3 卷积、步长 1 和 padding 1。单层编码器使用前置 LayerNorm、GELU 和零 Dropout。两个框架的注意力均包含三个独立的 Q/K/V 投影，并按矩阵乘、Softmax、矩阵乘计算。四个模型都使用平均交叉熵。

每个训练步包含清空梯度、前向、损失、反向和参数更新。初始化后先执行一次，再预热 200 步，随后测量 7 组，每组 10 步，取组平均耗时的中位数。各实现分别进行三轮测量，表中再取三轮的中位数，Ctorch 前后版本交替运行。计时包含宿主提交与等待 GPU 完成；每步读取损失、文件读写和参数初始化不在计时范围内。测量进程固定在同一个逻辑处理器上。

| 模型 | Ctorch 优化前 / ms | Ctorch 当前 / ms | PyTorch eager / ms | PyTorch 三轮范围 / ms |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.441 | 0.441 | 0.549 | 0.548–0.640 |
| 卷积加大分类头 | 2.367 | 2.371 | 4.806 | 4.795–4.809 |
| 三层卷积加全局平均池化 | 1.881 | 1.661 | 1.569 | 1.524–1.576 |
| 单层编码器 | 1.289 | 1.288 | 2.784 | 2.174–2.936 |

Ctorch 的前后两列使用相同的模型、初值和输入，均已启用 Linear 融合、梯度缓冲转交、批量参数更新与显存复用。当前实现将较大的自适应平均池化窗口交给一个线程块协作处理。三层卷积网络的完整训练步耗时减少约 11.7%，三轮配对的降幅为 11.4%–15.1%；其他三种模型基本持平。四种模型的在用显存峰值均不变。实现和瓶颈分析见[性能与内存](../performance.md#并行处理较大的池化窗口)。

表中三个实现都在同一轮测量中运行，GPU 没有其他计算任务。这里的优化幅度只使用本轮的 Ctorch 前后两列计算。不同负载下的历史计时不能直接作为对照。

这里使用固定输入重复更新参数来测量计算耗时。模型的数据集训练结果见前文和各项目页面。PyTorch 使用 eager 执行和默认优化器实现。

## CPU 前向耗时

在 Intel Core Ultra 9 275HX、Windows 11 上依次运行三个框架。CPU 算子内与算子间线程数设为 1，输入已在内存中。Ctorch 使用 GCC 16.1 Release 与 OpenBLAS 0.3.34；PyTorch 使用 2.11.0，TensorFlow 使用 2.20.0。

测量使用上节的批量和模型，在一次 SGD 更新后计时。先预热 3 次，再运行 7 组、每组 5 次前向，取每组平均耗时的中位数。计时包含模型调用，不包含读取文件、载入权重、初始布局转换、反向传播或参数更新。

Transformer 的 Ctorch 耗时取三轮独立测量的中位数。

| 模型 | Ctorch 普通分配 / ms | Ctorch 缓冲复用 / ms | PyTorch / ms | TensorFlow / ms |
| --- | ---: | ---: | ---: | ---: |
| MLP，批量 32 | 0.064 | 0.064 | 0.059 | 0.234 |
| LeNet，批量 16 | 0.797 | 0.342 | 0.523 | 0.736 |
| ResNet-20，批量 4 | 11.920 | 4.615 | 3.998 | 9.429 |
| Transformer，批量 2、长度 128 | 11.149 | 8.166 | 5.734 | 13.713 |

“缓冲复用”在计算外使用 `dl::MemoryGuard`，重复调用可以借用此前释放的存储。两种 Ctorch 配置使用相同内核、权重和输入，输出、梯度和 SGD 更新逐元素相同。写法和内存生命周期见[性能与内存](../performance.md)。

在这个配置下，启用缓冲复用的 Ctorch LeNet 前向耗时低于 PyTorch，ResNet-20 约为 PyTorch 的 1.15 倍，Transformer 约为 1.42 倍。批量、布局、设备和计算后端都会影响耗时。

PyTorch 和 TensorFlow 均直接使用 eager 算子，没有开启 `torch.compile` 或 `tf.function`。PyTorch 的 Python 环境为 3.14，TensorFlow 为 3.12。Ctorch/PyTorch 的图像布局是 NCHW，TensorFlow 使用 NHWC，布局转换在比较程序载入数据时完成。三个 Transformer 都使用普通矩阵运算实现因果注意力。

三框架的这张表统一使用 CPU。TensorFlow 2.20 的原生 Windows 包不提供 CUDA GPU 支持；运行其 GPU 版本可以按[官方安装说明](https://www.tensorflow.org/install/pip)使用 WSL2 或 Linux。

## Transformer 连续参数更新

使用同一组 Transformer 权重和同一批文本输入，三个框架连续进行 24 次 SGD 更新，学习率为 0.001。Dropout 关闭，前向、反向和参数更新均正常执行。每一步比较交叉熵，最后比较全部可训练参数。

每个训练步包含清空梯度、前向、交叉熵、反向和 SGD 更新。前 3 步预热，随后 21 步按每组 3 步计算平均耗时，再取 7 组的中位数。硬件、精度和线程数与前向比较相同，Ctorch 启用 `MemoryGuard`。

| 框架 | 每步耗时 / ms | 24 步交叉熵最大误差 | 最终参数最大误差 |
| --- | ---: | ---: | ---: |
| Ctorch | 20.956 | — | — |
| PyTorch | 18.308 | 8.34e-7 | 1.19e-7 |
| TensorFlow | 59.611 | 7.16e-7 | 1.19e-7 |

误差以 Ctorch 为参照。这个配置下，Ctorch 的训练步耗时约为 PyTorch 的 1.14 倍。

启用训练模式并将 Dropout 设为 0.1 时，Ctorch 的完整训练步耗时为 29.930 ms。Dropout 会增加掩码生成、输入缩放和反向梯度缩放的计算。

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
