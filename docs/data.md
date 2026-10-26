# 数据加载与变长批次

`DataLoader` 决定读取哪些样本。`Collate` 把这些样本整理成一个批次。`PrefetchLoader` 负责提前整理和传输批次。数据可以来自内存、文件或应用自己的读取函数。

## 多输入批次

每个 `Batch` 包含 `input`、`target` 和 `extra`。`extra` 按顺序存放其他输入，例如文本 mask、序列长度或第二路图像。

```cpp
dl::data::Batch batch{images, labels, {mask, positions}};
auto logits = model(batch.input, batch.extra[0], batch.extra[1]);
```

`extra` 中的张量与前两项一起传输，并随预取检查点保存。模型调用仍由普通 C++ 函数表达，输入的含义由应用确定。

## 填充变长序列

`pad_sequence` 接收一个非空张量列表。各张量的第一维是长度，其余维度一致。返回值包含填充后的数据、原始长度和有效位置 mask。

```cpp
auto a = dl::from_span<dl::Index>(std::vector<dl::Index>{4, 8, 2}, {3});
auto b = dl::from_span<dl::Index>(std::vector<dl::Index>{6}, {1});
std::vector<dl::Tensor> samples{a, b};
auto padded = dl::data::pad_sequence(samples, 0);
// values: [[4,8,2], [6,0,0]]，shape 为 [2,3]。
// lengths: [3,1]。
// mask: [[true,true,true], [true,false,false]]。
```

输入为 `[length, features]` 时，输出为 `[batch, max_length, features]`。浮点输入保留梯度，填充位置不回传到原序列。单个序列可以为空。

填充后的平均值需要排除无效位置：

```cpp
auto mask = padded.mask.astype(dl::DType::f32).unsqueeze(-1);
auto embeddings = embedding(padded.values);
auto pooled = dl::sum(embeddings * mask, {1}) / dl::sum(mask, {1});
```

这里每条文本至少包含一个字节。mask 为 `true` 的位置参与平均，填充位置对 Embedding 梯度的贡献为零。完整读取和训练代码见[变长文本分类](projects/variable-length.md)。

## 按长度分桶

长度相近的样本放在同一批次中，可以减少填充。`bucket_order` 先打乱样本，再在局部窗口内按长度排序，最后打乱完整批次。末尾不足一批的样本保留在末尾。

```cpp
dl::Generator sampling(42);
auto order = dl::data::bucket_order(lengths, 32, sampling);
loader.reset(std::move(order));
```

`lengths[i]` 是第 `i` 个样本的长度。默认窗口包含 32 个批次，可通过第四个参数调整。分桶只改变读取顺序，不改变样本本身。

## 加权采样与子集

`weighted_order` 按每个样本的非负权重做有放回采样。零权重样本不被抽取。至少一个样本的权重大于零。

```cpp
auto order = dl::data::weighted_order(sample_weights, 1000, sampling);
loader.reset(std::move(order));
```

类别不均衡时，可以令每个样本的权重等于其类别样本数的倒数。一次采样可能多次选中同一个样本。

`reset(order)` 也接受自定义子集或排列。随后 `next()` 按这份列表迭代，列表长度决定本轮样本数。无参数的 `reset()` 则恢复完整数据集，并按 `LoaderOptions::shuffle` 选择是否打乱。

## 后台整理和设备预取

```cpp
dl::data::DataLoader source(sample_count, collate, {.batch_size = 32}, sampling);
dl::data::PrefetchLoader loader(source, device, 2, true);
while (auto batch = loader.next()) {
    auto logits = model(batch->input);
    // 计算损失、反向传播和参数更新。
}
```

最后一个参数 `true` 启用单个后台 CPU 线程。该线程在模型计算时整理下一批数据，设备传输由调用线程提交。第三个参数控制设备预取深度，此外最多有一个正在整理或已整理的 CPU 批次。默认不开启后台线程。

数据整理函数返回拥有本批数据的 CPU 张量。数据增强使用单独的 `Generator`，不与模型初始化或 dropout 共用。后台运行期间通过预取器调用 `next()` 和 `reset()`，源 `DataLoader` 由它独占。

CUDA 预取复用页锁定暂存缓冲。直接返回页锁定张量时，批次拥有其存储，传输完成前不修改该存储。

## 保存和恢复

```cpp
std::array<dl::io::NamedGenerator, 1> generators{{{"augmentation", augmentation}}};
auto checkpoint = dl::io::capture_checkpoint(model, optimizer, progress, loader, generators);
dl::io::save_checkpoint("latest.train.ct", checkpoint);

auto saved = dl::io::load_checkpoint("latest.train.ct");
progress = dl::io::restore_checkpoint(saved, model, optimizer, loader, generators);
```

采集检查点时，预取器先完成正在进行的整理，再保存待交付批次、读取位置和随机状态。恢复后先使用这些批次，因此已经执行的数据增强不会重做。下一次 `next()` 自动继续后台整理。

训练检查点使用格式 7，读取器兼容格式 1–7。附加批次张量和后台整理配置随检查点保存。

手动保存预取状态时，先调用 `loader.state_dict()`，随后读取 `loader.source().state_dict()` 和增强生成器状态。这样三部分来自同一个读取位置。常规训练直接使用上述完整检查点接口即可。
