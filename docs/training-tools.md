# 训练控制与评价指标

`<dl/training.h>` 提供早停、平台期学习率调整、参数指数移动平均和混淆矩阵。它们也由 `<dl/dl.h>` 引入。训练循环仍由应用控制，验证结束后传入指标即可。

## 早停

```cpp
dl::training::EarlyStopping stopping({
    .mode = dl::training::MetricMode::minimize,
    .patience = 5,
    .min_delta = 1e-4,
});
// 每轮验证结束后调用一次。
if (stopping.step(validation_loss)) break;
```

指标比最佳值改善超过 `min_delta` 时，未改善次数清零。连续 5 次未改善后，`step()` 返回 `true`。比较准确率时把模式设为 `maximize`。输入指标是有限数值，`patience` 为正整数。

`best()` 返回最近一次超过 `min_delta` 的改善值，用于早停判断。保存验证指标最优的模型时，应用另行维护最优指标，每次变好就保存权重，见[分割示例](projects/segmentation.md)。

## 平台期调整学习率

```cpp
dl::training::ReduceLROnPlateau scheduler(optimizer, {
    .monitor = {.patience = 2, .min_delta = 1e-4},
    .factor = .5,
    .minimum_rate = 1e-6,
});
scheduler.step(validation_loss);
```

连续未改善次数超过 `patience` 时，每个参数组的当前学习率乘以 `factor`，并清零计数。`patience=2` 表示第三次未改善时降低学习率；`patience=0` 表示首次未改善就降低。学习率不会降到 `minimum_rate` 以下。

这个调度器在验证后调用。`StepLR` 等按更新次数推进的调度器继续使用原有 `step()` 接口。

## 参数指数移动平均

```cpp
dl::training::ExponentialMovingAverage ema(model, .999);
// 每次参数更新后执行。
optimizer.step();
ema.update(model);

// evaluation 与 model 具有相同结构。
ema.copy_to(evaluation);
evaluation.eval();
```

每次更新计算 `average = decay × average + (1-decay) × parameter`。初始平均值是构造时的模型参数。参数使用累积精度保存，低精度参数对应 FP32 平均值；BatchNorm 运行统计等 buffer 直接复制本次模型值。

评估模型与训练模型分开保存，评估不会覆盖优化器正在更新的参数。`copy_to()` 按目标模型的参数类型写入平均值。

## 分类和分割指标

```cpp
dl::training::ConfusionMatrix metric(classes);
metric.update(dl::argmax(logits, 1), labels, -100);
auto scores = metric.scores();
std::cout << scores.accuracy;
```

输入是同形状的整数类别编号。`ignore_index` 可用于跳过填充标签。矩阵的行是真实类别，列是预测类别。返回的 `precision`、`recall`、`f1` 和 `iou` 各包含一个类别的指标。

二值分割先把 logits 与零比较，再转换为 `i64`。每个像素作为一个样本累计。分母为零的类别比率取零；宏平均可以对所选类别的比率取平均。需要忽略背景时，只平均前景类别。

指标在 CPU 上累计整数计数。对 CUDA 张量调用 `update()` 会读取类别编号，因此通常在验证阶段或按固定间隔更新。

## 随检查点保存

```cpp
auto state = dl::io::capture_checkpoint(model, optimizer, progress, loader);
dl::io::set_extra(state, "ema", ema.state_dict());
dl::io::set_extra(state, "stopping", stopping.state_dict());
dl::io::set_extra(state, "plateau", scheduler.state_dict());
dl::io::save_checkpoint("latest.train.ct", state);

auto saved = dl::io::load_checkpoint("latest.train.ct");
progress = dl::io::restore_checkpoint(saved, model, optimizer, loader);
ema.load_state_dict(dl::io::get_extra(saved, "ema"));
stopping.load_state_dict(dl::io::get_extra(saved, "stopping"));
scheduler.load_state_dict(dl::io::get_extra(saved, "plateau"));
```

`set_extra()` 以指定前缀保存独立 CPU 副本，同名前缀再次写入时替换旧状态。工具的配置和计数一同保存，学习率随优化器恢复。前缀应互不包含，例如分别使用 `ema`、`stopping` 和 `plateau`。

混淆矩阵通过单个整数张量保存：`metric.state_dict()`。需要恢复一轮中间的评价指标时，把该张量放入 `TrainingCheckpoint::extra`，再调用 `metric.load_state_dict()`。
