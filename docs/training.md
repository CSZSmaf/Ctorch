# 模型与训练

## 模块

`Module` 登记参数、buffer 和子模块，提供 `parameters()`、`train()` / `eval()`、设备迁移和状态表。共享参数只更新一次；BatchNorm 的运行统计、位置索引和固定掩码可登记为 buffer。

常用层包括 Linear、Conv1d/2d、固定与自适应池化、Upsample、BatchNorm、GroupNorm、LayerNorm、RMSNorm、Embedding、Dropout、RNN/GRU/LSTM、MultiHeadAttention 和 TransformerEncoder。自定义模块通过普通 C++ 运算组合已有层，见 [custom_layer.cpp](../examples/custom_layer.cpp)。

在构造优化器前迁移模型，使参数与优化器状态具有一致设备。

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

cuDNN 低精度卷积先按选定的 16 位格式写回，再展开为 FP32。融合注意力与组合注意力的计算次序和低精度舍入不同，不能假定逐位一致。注意力后端通过选项显式选择，默认使用组合实现。

## 保存与恢复

轻量权重文件保存参数和 buffer。训练快照另保存优化器槽、参数分组、调度器、随机流、数据排列与游标、精度及缩放状态；预取快照还保存已整理但未消费的批次。

快照按规范参数名称恢复状态，支持模型参数的登记次序变化。保存和恢复安排在更新边界；微批窗口内的中间反向图不写入文件。

真实训练示例共用训练、验证、最佳模型选择和恢复逻辑，命令见[示例说明](../examples/README.md)。
