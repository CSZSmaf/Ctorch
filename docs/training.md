# 模型与训练

## 模块

`Module` 登记参数、buffer 和子模块，提供 `parameters()`、`train()` / `eval()`、设备迁移和状态表。共享参数只更新一次；BatchNorm 的运行统计、位置索引和固定掩码可登记为 buffer。

常用层包括 Linear、Conv1d/2d、固定与自适应池化、Upsample、BatchNorm、GroupNorm、LayerNorm、RMSNorm、Embedding、Dropout、RNN/GRU/LSTM、MultiHeadAttention 和 TransformerEncoder。自定义模块通过普通 C++ 运算组合已有层，见 [custom_layer.cpp](../examples/custom_layer.cpp)。

带参数的层在构造时使用 `register_parameter` 登记，在前向中通过 `bound(parameter)` 读取当前绑定；buffer 对应 `register_buffer` 和 `bound_buffer`。这样同一个前向既能使用模块自身状态，也能使用 `FunctionalModule` 提供的独立状态。完整模板见 [parameter_layer.cpp](../examples/parameter_layer.cpp)。

参数、buffer 与子模块共用本层名称空间，名称非空、不含点且唯一。`dl::summary(model)` 展示模块路径、参数量、冻结状态和共享关系；`dl::describe(tensor)` 展示布局与设备等元信息，不读取设备数值。

在构造优化器前迁移模型，使参数与优化器状态具有一致设备。

## 分类输出与损失

每个样本只属于一个类别时，模型输出各类别的 logits，标签使用 `i64` 类别编号，损失使用 `cross_entropy`。

二分类和多标签分类使用 `binary_cross_entropy_with_logits`。每个输出表示一个独立类别的分数，目标与输出同形，采用 `[0,1]` 内的浮点值。二分类可以使用 `[batch, 1]`，多标签分类可以使用 `[batch, classes]`，逐像素分类可以使用 `[batch, classes, height, width]`。

```cpp
auto logits = model(input);
auto loss = dl::binary_cross_entropy_with_logits(logits, targets);
loss.backward();

// 预测时用 sigmoid 把每个独立分数转成概率。
auto probabilities = dl::sigmoid(logits.detach());
```

训练时直接传入 logits。该损失在内部计算稳定的对数表达式，无需先调用 sigmoid。默认 `mean` 对所有损失项取平均；`none` 返回逐项损失，可用于样本加权。目标也可以是软标签，例如 `0.8`。

## 更新与控制

优化器包括 SGD、Adam、AdamW、RMSprop 和 Adagrad。参数分组各自持有学习率与权重衰减，可配合冻结、解冻、梯度裁剪和学习率调度。`zero_grad()` 清空叶梯度槽，多个微批的反向可在同一次更新前累计。

一个训练步依次完成梯度清理、前向、损失、反向及优化器更新。微批损失按有效样本或有效标签数加权，最后一个不足整批的窗口使用实际权重。评估使用 `eval()` 和无求导作用域。

分阶段微调见 [finetune.cpp](../examples/finetune.cpp)，循环网络训练见 [recurrent_train.cpp](../examples/recurrent_train.cpp)。

## 数据

DataLoader 管理采样排列、游标和批次整理函数。预取器采用有界缓冲；CUDA 输入可使用页锁定宿主存储和异步传输。传输对象保留源、目标及完成事件，消费端在正确 stream 上接收数据。

数据集由应用拥有。MNIST 示例读取解压后的 IDX 文件，文本示例将字节映射为类别；库加载和示例构建不自动下载数据。

## 混合精度

`AutocastGuard` 为选定算子指定 f16 或 bf16 计算输入。参数、梯度和优化器状态保留 FP32；FP16 训练可配合 `GradScaler` 缩放损失、还原梯度、跳过非有限更新及调整缩放因子。

裁剪安排在梯度还原之后。恢复训练时同时恢复精度配置、缩放器和更新计数，使微批窗口与参数更新保持对应关系。

`scaler.step(optimizer)` 组合梯度还原、条件更新和倍率调整。需要裁剪时，调用一次 `scaler.unscale_(optimizer)`，按其有限性返回值裁剪并执行 `optimizer.step()`，最后调用 `scaler.update(finite)`。此时更新流程已经显式完成。

cuDNN 低精度卷积先按选定的 16 位格式写回，再展开为 FP32。融合注意力与组合注意力的计算次序和低精度舍入不同，不能假定逐位一致。注意力后端通过选项显式选择，默认使用组合实现。

## 保存与恢复

轻量权重文件保存参数和 buffer。训练快照另保存优化器槽、参数分组、调度器、随机流、数据排列与游标、精度及缩放状态；预取快照还保存已整理但未消费的批次。

`state_dict()` 返回共享当前存储的状态表。需要保留最佳模型的内存快照时，使用 `dl::clone_state(model.state_dict())` 创建独立、无求导历史的 CPU 副本；第二个参数可指定其他保存设备。训练快照接口本身也采集独立副本。

共享张量采用全部登记路径中字典序最小的规范名，参数槽仍保留首次遇到该身份的顺序。快照按名称恢复模型与优化器状态，支持名称集合相同、登记次序不同的模块树；旧检查点中仍存在于当前树的共享别名也可加载。`state_aliases()` 提供别名到规范名的对应。路径改名由应用明确迁移。

保存和恢复安排在更新边界；微批窗口内的中间反向图不写入文件。

真实训练示例共用训练、验证、最佳模型选择和恢复逻辑，命令见[示例说明](../examples/README.md)。
