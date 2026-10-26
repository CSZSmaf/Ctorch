# 损失函数

模型输出与目标的含义决定损失函数。分类输出通常使用 logits，即尚未归一化的类别分数。回归输出直接表示预测数值。

| 任务 | 函数 | 输入与目标 |
| --- | --- | --- |
| 单标签分类、逐像素多类分割 | `cross_entropy` | logits `[N,C,...]`，整数标签 `[N,...]` |
| 二分类、多标签分类 | `binary_cross_entropy_with_logits` | 同形 logits 和 `[0,1]` 目标 |
| 难易样本不均衡的二分类 | `focal_loss` | 同形 logits 和 0/1 标签 |
| 二值分割 | `dice_loss` | 同形概率和掩码 |
| 分布匹配、知识蒸馏 | `kl_div` | 对数预测概率和目标分布 |
| 数值回归 | `mse_loss`、`l1_loss`、`smooth_l1_loss`、`huber_loss` | 可广播的预测与目标 |

## 归约方式

`LossReduction::none` 返回每项损失，`sum` 返回总和，`mean` 返回平均值。BCE、Focal 和 KL 的 `mean` 按全部损失元素数取平均。Dice 先沿指定轴计算每组损失，再执行归约。

交叉熵的加权平均使用有效标签对应的类别权重之和作为分母。比较不同框架时，应使用相同的归约方式。

## 类别权重和标签平滑

```cpp
dl::CrossEntropyOptions options;
options.weight = dl::from_span<float>(std::vector<float>{1, 2, .5f}, {3}, false, device);
options.label_smoothing = .1;
options.ignore_index = -100;
auto loss = dl::cross_entropy(logits, labels, options);
```

这里三类的权重分别为 1、2、0.5。权重是一维非负张量，类型为 logits 的累积类型，设备与 logits 一致。目标编号为 `-100` 的位置不参与计算。

平滑系数为 ε、类别数为 C 时，目标分布是 `(1-ε) × one_hot(label) + ε/C`。例如三分类、标签为第一类、ε 为 0.1 时，分布约为 `[0.9333, 0.0333, 0.0333]`。类别权重同时作用于各类别的目标项，平均分母仍为有效原始标签的权重之和。这与 PyTorch 的整数标签加权交叉熵约定一致。

单个样本可以使用 `[C]` logits 和标量标签。类别权重作为固定系数，不求导。

## 二分类中的权重

```cpp
dl::BinaryLossOptions options;
options.positive_weight = dl::full({}, 3, {.device = device});
auto loss = dl::binary_cross_entropy_with_logits(logits, target, options);
```

`positive_weight` 只放大正类项；`weight` 放大整个逐元素损失。两者都可广播到损失形状。例如输出为 `[N,C,H,W]` 时，每类正权重可使用 `[C,1,1]`。BCE 在内部使用稳定的 Softplus，不需要先调用 sigmoid。

## Focal 与 Dice

```cpp
auto focal = dl::focal_loss(logits, mask, 2.0, .25);
auto dice = dl::dice_loss(dl::sigmoid(logits), mask, {2, 3});
auto loss = focal + dice;
```

Focal 的 `gamma` 控制简单样本的衰减程度。`gamma=0` 时得到带正负类别系数的 BCE。`alpha` 是正类系数，负类系数为 `1-alpha`。目标是 0/1 标签。

Dice 比较预测概率与目标掩码的重叠程度。对于 `[N,1,H,W]` 输入，沿 `{2,3}` 归约得到每张图的损失，再对批次取平均。它使用 `(2 × intersection + epsilon) / (prediction_sum + target_sum + epsilon)` 计算相似度，损失为一减相似度。预测和目标均为零时，损失为零。

分割示例把加权 BCE 与 Dice 相加，见[二值分割项目](projects/segmentation.md)。

## KL 散度与蒸馏

```cpp
auto log_student = dl::log_softmax(student_logits / temperature, 1);
auto teacher = dl::softmax(teacher_logits / temperature, 1).detach();
auto loss = dl::kl_div(log_student, teacher, dl::LossReduction::sum)
    / student_logits.size(0) * (temperature * temperature);
```

默认目标是非负概率，零概率项贡献零。目标已经是对数概率时，把第四个参数设为 `true`。`mean` 按全部元素取平均；按批次取平均时使用 `sum` 后除以批量大小。教师输出通过 `detach()` 固定，梯度只更新学生网络。

参考：[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html)、[BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html)、[KLDivLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.KLDivLoss.html)。
