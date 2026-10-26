# 二值图像分割

这个项目生成带噪声的圆形图像，训练一个小型编码器与解码器，为每个像素预测前景概率。它展示分割损失、跳跃连接、前景 IoU、EMA 评估和训练状态保存。

数据由程序生成：48 张图用于训练，16 张图用于验证，使用不同的随机种子。每张图为 `16×16`。这是合成图像示例，训练结果衡量模型对这类图形的分割能力。

## 运行

先按[项目构建说明](README.md#构建程序)启用示例，然后运行：

```sh
../Ctorch-work/build/bin/ctorch_segmentation ../Ctorch-work/segmentation cpu
```

CUDA 构建使用：

```sh
../Ctorch-work/build-gpu/bin/ctorch_segmentation ../Ctorch-work/segmentation-cuda cuda
```

程序训练最多 6 轮，打印训练损失、验证损失和前景 IoU。输出目录包含 `best.ct` 和 `latest.train.ct`，分别保存验证损失最低的 EMA 权重和继续训练所需的状态。

## 模型结构

| 步骤 | 输出形状 | 作用 |
| --- | --- | --- |
| 输入 | `[B,1,16,16]` | 灰度图像 |
| 3×3 卷积与 ReLU | `[B,4,16,16]` | 提取浅层特征，保存为 skip |
| 2×2 最大池化 | `[B,4,8,8]` | 降低空间分辨率 |
| 3×3 卷积与 ReLU | `[B,8,8,8]` | 组合较大范围的特征 |
| 双线性上采样 | `[B,8,16,16]` | 恢复原图尺寸 |
| 与 skip 拼接 | `[B,12,16,16]` | 合并浅层和深层特征 |
| 3×3 卷积、ReLU、1×1 卷积 | `[B,1,16,16]` | 输出逐像素 logits |

拼接使用 `concat({up, skip}, 1)`，发生在通道维。输出与输入图像具有相同空间尺寸。

## 损失与指标

目标掩码为 0/1 浮点张量。训练损失是加权 BCE 与 Dice 之和：

```cpp
dl::BinaryLossOptions options;
options.positive_weight = dl::full({}, 3, {.device = device});
auto loss = dl::binary_cross_entropy_with_logits(logits, target, options)
    + dl::dice_loss(dl::sigmoid(logits), target, {2, 3});
```

正类像素较少，因此 BCE 中的正类项乘以 3。Dice 衡量整幅掩码的重叠程度。两部分的定义见[损失函数](../objectives.md)。

验证时将 logits 大于零的位置判为前景，这等价于概率大于 0.5。`ConfusionMatrix(2)` 累计所有验证像素，`scores().iou[1]` 给出前景交并比。IoU 为前景交集像素数除以前景并集像素数。

## EMA 与恢复

每次 AdamW 更新后调用 `ema.update(model)`。验证前把平均参数复制到另一个同结构模型，并设置为评估模式。EMA 初值来自训练模型，衰减系数为 0.9。

训练检查点同时保存优化器、预取批次、EMA、早停和平台期调度器的状态。恢复这些工具的方法见[训练控制](../training-tools.md#随检查点保存)。更换为真实图像时，保留模型和训练步骤，在 `Collate` 中读取图像与对应掩码即可。几何增强应对图像和掩码使用相同变换。

完整代码：[segmentation.cpp](../../examples/segmentation.cpp)。
