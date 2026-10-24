# Shakespeare：Transformer

本项目用 Shakespeare 文本训练一个语言模型。模型读取一段文本的字节，预测每个位置后面的字节。训练后可以输入一段开头，逐步采样生成后续文本。

## 文本与序列

```sh
python examples/download_data.py shakespeare ../Ctorch-work/data/shakespeare
```

数据来自 [TensorFlow 文本生成教程](https://www.tensorflow.org/text/tutorials/text_generation)使用的公开 Shakespeare 文本，共 1,115,394 字节。示例先把原始文件按 90% / 5% / 5% 划分为训练、验证和测试区间，再在各区间内建立序列。

词表是 0 到 255 的字节值，不需要额外分词器。UTF-8 汉字通常占多个字节，因此这里的一个 token 是一个字节。英文文本中的多数可见字符只占一个字节。

每条输入最多包含 128 个字节，标签向后移动一个位置：

```text
原始片段：ROMEO:
输入字节：R O M E O
目标字节：O M E O :
```

[text_data.h](../../examples/text_data.h) 按 128 个位置切分序列。末尾不足长度时补齐输入，并将无效标签设为 `-100`。交叉熵忽略这些标签。输入和目标形状都是 `[N,128]`。

## 模型结构

模型包含 4 层因果 Transformer，每层宽度为 128，4 个注意力头，前馈隐藏宽度为 512。总计 842,496 个可训练参数。

| 部分 | 作用 |
| --- | --- |
| 256×128 字节 Embedding | 将字节编号转成向量 |
| 128×128 位置 Embedding | 表示序列中的位置 |
| 4 层 Transformer | 根据前文生成每个位置的特征 |
| 共享词表投影 | 把 128 维特征转成 256 个字节分数 |

每层使用 pre-LayerNorm、4 头注意力、GELU 前馈网络、残差连接和 0.1 的 dropout。因果掩码使当前位置只能读取自己和此前的位置。输入 Embedding 与输出投影共用同一份权重。

模型定义在 [project_models.h](../../examples/project_models.h) 的 `ByteTransformer` 中：

```cpp
examples::ByteTransformer model(examples::medium_text);
auto logits = model(input);  // [N,128,256]
auto loss = dl::cross_entropy(logits.reshape({-1,256}), target.reshape({-1}),
                              dl::LossReduction::mean, -100);
```

交叉熵接收 `[样本数,类别数]`，因此先把批次和位置合并成一维。忽略标签 `-100` 不计入损失平均值。

`ctorch_byte_transformer` 保留了宽度 32、2 层、长度 32 的小型配置。这里使用 `ctorch_shakespeare_transformer` 运行宽度 128 的配置。加载权重时要选择对应结构。

## 训练

```sh
../Ctorch-work/build/bin/ctorch_shakespeare_transformer ../Ctorch-work/data/shakespeare/input.txt ../Ctorch-work/runs/shakespeare --steps 2000 --batch-size 16 --eval-every 400 --clip-grad-norm 1
```

GPU 构建使用 `build-gpu/bin` 中的程序并追加 `--device 0`。优化器为 AdamW，学习率 0.001，权重衰减 0.01。字节和位置 Embedding 以标准差 0.02 的正态分布初始化。每 400 步评估一次，梯度范数裁剪上限为 1。

显存不足时可以追加 `--micro-batch-size 4`。每个微批次计算梯度后按有效标签数加权累计，完成整个批次后再更新参数。FP16/BF16 可通过 `--precision` 选择；FP32 默认使用普通数学注意力路径。

## 实际训练结果

这次运行使用 FP32、批量 16 和 2,000 次更新，在 NVIDIA RTX 5090 Laptop GPU 上完成。最终加载验证损失最低的模型，对测试区间的 55,769 个有效目标字节评估。

| 指标 | 结果 |
| --- | ---: |
| 初始验证交叉熵 | 5.60455 |
| 最佳验证交叉熵 | 1.73814 |
| 测试交叉熵 | 1.80701 |
| 测试困惑度 | 6.09221 |
| 下一字节准确率 | 46.20% |

交叉熵使用自然对数。困惑度由平均交叉熵计算：

<span data-tex="\operatorname{PPL}=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1}\mid x_{\le t})\right)">PPL = exp(平均交叉熵)</span>

这里的困惑度按字节计算，比较其他模型时需要使用相同文本划分和 token 定义。

## 生成文本

[text_generate.cpp](../../examples/text_generate.cpp) 加载训练权重，用当前前缀预测下一个字节，然后把采样结果追加到前缀。前缀超过 128 字节时只保留最近 128 字节。

```sh
../Ctorch-work/build/bin/ctorch_text_generate ../Ctorch-work/runs/shakespeare/best.ct "ROMEO:" --model medium --tokens 240 --temperature 0.8
```

GPU 程序追加 `--device 0`。`--tokens` 是新增字节数。温度将 logits 除以指定值后再计算 softmax；较小的正温度使分布更集中。生成时关闭 dropout 和梯度计算，采样种子固定为 42。

下面是这次模型实际生成的开头：

```text
ROMEO:
But again to
you and what young vilainess have may his mestands,
And he corturish his diess be see the connt.
```

模型已学到部分单词、换行和句式，仍会生成拼写错误。可以增加训练步数，比较验证困惑度和生成文本的变化。

## 保存与恢复

训练状态包括共享参数、AdamW 动量、随机状态和采样位置。继续到第 4,000 步：

```sh
../Ctorch-work/build/bin/ctorch_shakespeare_transformer ../Ctorch-work/data/shakespeare/input.txt ../Ctorch-work/runs/shakespeare --steps 4000 --batch-size 16 --eval-every 400 --clip-grad-norm 1 --resume ../Ctorch-work/runs/shakespeare/latest.train.ct
```

使用与原训练一致的设备、精度及批量设置。训练入口为 [shakespeare_transformer.cpp](../../examples/shakespeare_transformer.cpp)，共用流程在 [text_training.h](../../examples/text_training.h)。下一步可以修改层数或宽度，并使用新的输出目录训练。[框架对比](comparison.md)展示同一模型在三个框架中的前向和梯度结果。
