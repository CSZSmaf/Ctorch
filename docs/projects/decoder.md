# 字节解码器：训练与缓存生成

这个项目用 RoPE、分组查询注意力和 SwiGLU 训练下一字节预测模型，随后用 KV 缓存生成文本。它复用[文本训练流程](transformer.md#文本与序列)，增加独立的解码状态。

## 数据与模型

模型有两层，宽度 64，4 个 query 头和 2 个 K/V 头，SwiGLU 中间宽度 256，上下文容量 128。输入 Embedding 与输出 Linear 共享 `256×64` 的词表权重，共 139,584 个可训练参数。

输入与目标使用 `[batch,128]` 的字节编号。文件按原始位置划分为 90% 训练、5% 验证和 5% 测试，然后分别构造序列。标签比输入向后移动一个字节，补齐位置使用 `-100`。模型输出 `[batch,128,256]` 的 logits。

```cpp
examples::ByteDecoder model;
auto logits = model(input);
auto loss = dl::cross_entropy(logits.reshape({-1, 256}), target.reshape({-1}),
                              dl::LossReduction::mean, -100);
```

模型定义见 [decoder_model.h](../../examples/decoder_model.h)。各层和缓存接口见[解码器手册](../decoder.md)。

## 构建与训练

从仓库根目录运行以下命令。构建目录、语料和训练输出都位于源码目录外。

```sh
cmake -S . -B ../Ctorch-work/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCTORCH_BUILD_EXAMPLES=ON
cmake --build ../Ctorch-work/build --target ctorch_decoder_train ctorch_decoder_generate
python examples/download_data.py shakespeare ../Ctorch-work/data/shakespeare
../Ctorch-work/build/bin/ctorch_decoder_train ../Ctorch-work/data/shakespeare/input.txt ../Ctorch-work/runs/decoder --steps 2000 --batch-size 8 --eval-every 200 --clip-grad-norm 1
```

CPU 命令适合检查流程。使用 GPU 时，按[构建手册](../build.md)启用 CUDA，并在训练命令后加 `--device 0`。加 `--precision bf16` 使用 BF16 矩阵乘；参数、输出和优化器状态仍为 FP32。当前解码器使用数学注意力路径，`--attention` 保持默认 `math`。

优化器为 AdamW，学习率 0.001。训练记录验证交叉熵和下一字节准确率，按验证损失选择 `best.ct`，结束后评估测试区间。`metrics.csv` 和 `summary.json` 的字段见[项目目录](README.md#阅读输出)。

## 恢复训练

`--stop-after` 在指定更新次数后保存完整状态。下次运行用 `--resume` 继续，`--steps` 指恢复后的总更新目标。

```sh
../Ctorch-work/build/bin/ctorch_decoder_train ../Ctorch-work/data/shakespeare/input.txt ../Ctorch-work/runs/decoder --steps 2000 --batch-size 8 --stop-after 500
../Ctorch-work/build/bin/ctorch_decoder_train ../Ctorch-work/data/shakespeare/input.txt ../Ctorch-work/runs/decoder --steps 2000 --batch-size 8 --resume ../Ctorch-work/runs/decoder/latest.train.ct
```

使用同一语料、模型结构和批量大小。检查点恢复模型、优化器、采样位置、随机状态、训练指标和精度配置。生成缓存属于单次推理请求，恢复训练后按需新建。

## 生成文本

```sh
../Ctorch-work/build/bin/ctorch_decoder_generate ../Ctorch-work/runs/decoder/best.ct "ROMEO:" 48
```

使用 GPU 时在末尾加 `cuda`。前缀非空，前缀字节数与生成字节数之和不超过 128。程序按最大 logit 选择下一个字节，生成规则为贪心选择。

首次调用把整个前缀送入 `decode`，建立各层的 K/V。随后每次只输入刚生成的一个字节，前面位置的投影和特征可以复用。输出始终对应本次新输入的位置。

字节模型直接输出字节。英文文本便于观察；多字节 UTF-8 文本可能在生成结束处截断一个字符。训练步数较少时输出通常仍有重复或无意义片段，需要结合验证损失判断学习情况。

## 缓存计算耗时

以下测量使用 NVIDIA RTX 5090 Laptop GPU、CUDA 13.2、cuDNN 9.21 和 Release 构建。两种路径使用相同的两层解码器、模型宽度 64、4 个 query 头、2 个 K/V 头及中间宽度 256，batch 为 1。

固定一组特征输入，从长度 32 的前缀开始，连续计算 64 个位置的预测特征。完整前缀路径每次重新计算此前位置；缓存路径首次处理 32 个位置，随后每次追加一个位置。计时包括首次前缀计算、后续 63 次追加和输出汇总，CUDA 在计时前后同步。两种路径预热后交错运行三轮，下表为中位数。

| 精度 | 每次重算完整前缀 | KV 缓存 | 耗时减少 |
| --- | ---: | ---: | ---: |
| FP32 | 72.59 ms | 52.88 ms | 27.2% |
| BF16 autocast | 79.38 ms | 64.43 ms | 18.8% |

两种精度下两条路径的最大绝对误差均为 `4.77e-7`。该测量针对解码器本体，不包含分词、Embedding、词表投影、采样或文件读取。BF16 在这个小模型中较慢，低精度转换与算子启动成本会影响结果。

CPU FP64 和 CUDA FP32 另使用相同参数对照 PyTorch 2.11 的矩阵乘、softmax、RMSNorm、SiLU 和 AdamW。对照覆盖前向、梯度及四步参数更新。缓存还比较了多 token 分块、单 token 追加、清空和截断后重新追加。

## 相关源码

- [训练入口](../../examples/decoder_train.cpp)
- [缓存生成](../../examples/decoder_generate.cpp)
- [共用文本训练流程](../../examples/text_training.h)
- [模型定义](../../examples/decoder_model.h)
