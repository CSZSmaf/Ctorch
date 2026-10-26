# 变长文本分类

这个项目从 TSV 文件读取文本，用 Embedding 和带 mask 的平均池化完成分类。每个批次按最长文本填充，并根据文本长度分桶。不同长度的文本可以放入同一个训练循环。

## 数据格式

每行包含整数标签、一个制表符和非空 UTF-8 文本。标签从零开始，范围小于类别数。训练文件与验证文件分别准备，例如二分类：

```text
0	这次体验很好
1	商品已经损坏
0	安装很方便
1	无法正常启动
```

上述标签含义由数据集确定。文件中的分隔符是实际制表符。文本按 UTF-8 字节编码，字节值加一后作为 token，零用作填充值。词表包含 257 项，无需额外分词库。

## 运行

```sh
../Ctorch-work/build/bin/ctorch_variable_length ../Ctorch-work/train.tsv ../Ctorch-work/validation.tsv 2 ../Ctorch-work/text-classifier cpu
```

最后一个参数改为 `cuda` 并使用 CUDA 构建即可在 GPU 上训练。程序打印验证损失和准确率，将验证损失最低的模型保存为 `best.ct`。

## 批次包含什么

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| `input` | `[B,L]` | 填充后的整数 token |
| `target` | `[B]` | 分类标签 |
| `extra[0]` | `[B,L]` | 有效位置 mask |
| `extra[1]` | `[B]` | 原始字节长度 |

`pad_sequence` 一次生成数据、长度和 mask。`PrefetchLoader` 将这些张量一起传到模型设备。后台整理只负责 CPU 批次，不执行模型计算。

每轮开始时，`bucket_order` 把长度相近的文本安排在同一批次中。这样可以减少填充位置。详细的采样和预取接口见[数据加载](../data.md)。

## 模型与训练

Embedding 把每个 token 转换为 32 维向量。模型先排除填充位置，再沿序列维取平均：

```cpp
auto mask = batch.extra[0].astype(dl::DType::f32).unsqueeze(-1);
auto features = dl::sum(embedding(batch.input) * mask, {1}) / dl::sum(mask, {1});
auto logits = classifier(features);
```

这个模型的平均特征不保留字节顺序，适合先检查数据读取和分类流程。需要表达顺序关系时，可以将平均池化替换为循环网络或注意力编码器，继续使用同一份长度和 mask。

训练使用 AdamW 和 0.05 的标签平滑。验证使用普通交叉熵，按照样本数累计平均损失。连续三轮验证损失没有改善时结束训练。

完整代码：[variable_length.cpp](../../examples/variable_length.cpp)。
