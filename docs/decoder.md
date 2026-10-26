# 解码器与低秩微调

`<dl/decoder.h>` 提供 RoPE、SwiGLU、LoRA、分组查询注意力和因果解码器。`<dl/dl.h>` 已包含这些声明。它们使用现有张量、自动求导和模块系统，参数保存、设备迁移与函数式调用沿用普通模块的接口。

## RoPE 位置编码

RoPE 按位置旋转 query 和 key 的特征。Ctorch 将相邻两个特征作为一对：

<span data-tex="y_{2i}=x_{2i}\cos(p\theta_i)-x_{2i+1}\sin(p\theta_i)">y[2i] = x[2i] cos(pθ) - x[2i+1] sin(pθ)</span>

<span data-tex="y_{2i+1}=x_{2i}\sin(p\theta_i)+x_{2i+1}\cos(p\theta_i),\quad\theta_i=b^{-2i/d}">y[2i+1] = x[2i] sin(pθ) + x[2i+1] cos(pθ)</span>

其中 `p` 是位置，`d` 是每个头的特征数，`b` 默认等于 10000。特征数为正偶数，输入倒数第二维是序列长度。

```cpp
dl::nn::RotaryEmbedding rotary(16, 128);
auto query = dl::randn({2, 4, 8, 16});
auto rotated = rotary.forward(query, 20); // 位置 20 到 27
```

余弦和正弦表登记为 buffer，随模型保存和迁移。`capacity` 是可使用的位置数。FP16/BF16 配置以 FP32 建表。已有角度表时，可以直接调用 `dl::rotary_embedding(input, cosine, sine)`；表的末维为 `d/2`，其余维度与输入的相邻特征对广播。

导入其他实现的权重时，要核对其旋转方式。有些实现把特征前半段与后半段配对，与这里的相邻配对需要不同的 Q/K 权重排列。

## SwiGLU 前馈层

SwiGLU 用一条投影分支控制另一条分支：

<span data-tex="\operatorname{SwiGLU}(x)=W_d\bigl(\operatorname{SiLU}(W_gx)\odot W_ux\bigr)">SwiGLU(x) = down(SiLU(gate(x)) × up(x))</span>

```cpp
dl::nn::SwiGLU feedforward(64, 256);
auto output = feedforward(dl::randn({2, 32, 64})); // [2,32,64]
```

构造参数依次为输入宽度、中间宽度、是否使用 bias 和张量选项。默认不使用 bias。`gate` 与 `up` 投影到中间宽度，`down` 投影回输入宽度。

## 分组查询注意力

普通多头注意力为每个 query 头提供对应的 key/value 头。分组查询注意力让多个相邻 query 头共享一组 key/value，减少 K/V 投影和缓存的大小。

```cpp
auto query = dl::randn({2, 8, 3, 16});
auto key   = dl::randn({2, 2, 20, 16});
auto value = dl::randn({2, 2, 20, 24});
auto output = dl::grouped_query_attention(query, key, value); // [2,8,3,24]
```

Q、K、V 都使用 `[batch,heads,length,features]`。query 头数是 K/V 头数的正整数倍，K 与 V 的头数及长度一致。上例中每四个相邻 query 头共享一个 K/V 头。

可选 `keep_mask` 是布尔张量，`true` 表示可读取的位置，形状可广播到 `[batch,query_heads,query_length,key_length]`。函数默认没有因果限制；`GroupedQueryAttention` 模块和解码器负责按位置构造因果掩码。当前实现由批量矩阵乘、softmax 与广播组成，使用 CPU 或 CUDA 后端，不选择 cuDNN 融合注意力。

## 训练解码器

```cpp
dl::nn::TransformerDecoder decoder(
    2,    // 层数
    64,   // 模型宽度
    4,    // query 头数
    2,    // key/value 头数
    256,  // SwiGLU 中间宽度
    128); // 上下文容量

auto input = dl::randn({2, 32, 64});
auto features = decoder(input); // [2,32,64]
```

每层依次执行 RMSNorm、带 RoPE 的因果注意力、残差相加、RMSNorm、SwiGLU 和残差相加。层堆叠后再做一次 RMSNorm，epsilon 为 `1e-5`。当前层没有 dropout。

输入是特征向量。语言模型在前面接 Embedding，在后面接词表 Linear，训练时对全部有效位置计算交叉熵。完整项目见[字节解码器](projects/decoder.md)。

## KV 缓存

训练调用普通 `forward`。生成时调用 `decode`，每次传入新增 token 的特征：

```cpp
decoder.eval();
auto cache = decoder.make_cache();
auto first = decoder.decode(dl::randn({1, 12, 64}), cache);
auto next = decoder.decode(dl::randn({1, 1, 64}), cache);
// 各层缓存现在有 13 个位置，next 的形状是 [1,1,64]。
```

缓存第一次使用时分配固定容量的 K/V 存储，之后把新位置写入空余部分。读取返回有效前缀的视图，避免反复拼接整个缓存。各层分别保存 K/V，RoPE 的起始位置取自已有长度。

一次调用也可以追加多个 token。新块中每个位置只能读取自己和此前的 key；单 token 调用可以读取全部已有 key。`decode` 内部关闭梯度记录。

一个请求持有一组缓存，并按顺序使用。`clear()` 清空长度并复用存储，`truncate(n)` 保留前 `n` 个位置。缓存视图共享存储，后续追加可能改写截断部分。更换模型权重、batch 大小、设备或 dtype 后，为请求重新调用 `make_cache()`。缓存不写入模型的 `state_dict()`。

缓存保存每个位置各自的上下文状态，不能只移动数据就当作重新计算的滑动窗口。达到容量后，可选取新的前缀并使用一组空缓存重新计算。

## LoRA 低秩微调

LoRA 冻结原 Linear，用两个较小的可训练矩阵表达权重增量：

<span data-tex="y=xW^T+b+\frac{\alpha}{r}xA^TB^T">y = Linear(x) + (α/r) up(down(x))</span>

```cpp
dl::nn::Linear base(64, 32);
dl::nn::LoRALinear adapter(base, 4, 8.0);
dl::optim::AdamW optimizer(adapter.parameters(), {.learning_rate = 0.001});
auto input = dl::randn({2, 64});
auto target = dl::zeros({2, 32});

optimizer.zero_grad();
auto loss = dl::mse_loss(adapter(input), target); // input[...,64], target[...,32]
loss.backward();
optimizer.step();

auto inference_layer = adapter.merged();
```

适配器持有基层参数的独立冻结副本，修改它不会改变传入的 `base`。`down` 形状为 `[rank,input_features]`，`up` 形状为 `[output_features,rank]`。`up` 从零开始，因此初始输出与基层相同。优化器跳过没有梯度的冻结参数。

`merged()` 返回合并增量后的独立、冻结 Linear，适合保存和推理。它使用参数精度计算合并，不受外层 autocast 影响。若参数本身为 FP16/BF16，合并权重仍需舍入到该类型。适配器保持原样，继续训练仍调用适配器。

`adapter.state_dict()` 保存基层副本及两组低秩参数，可使用普通权重文件或训练检查点恢复。模块替换由应用在构造模型时完成。

## C17

包含 `<dl/decoder_c_api.h>`，使用 `dl_transformer_decoder_create` 创建模块，训练沿用 `dl_module_forward` 和优化器接口。`dl_decoder_cache_create` 创建请求缓存，`dl_transformer_decoder_decode` 追加输入，`dl_decoder_cache_clear` / `truncate` 管理长度，最后调用 `dl_decoder_cache_release`。

该头文件也提供 `dl_tensordot`、RoPE、SwiGLU、GQA、LoRA 构造及合并入口。返回的模块和张量分别由 `dl_module_release`、`dl_tensor_release` 释放。

## 参考

- [RoFormer](https://arxiv.org/abs/2104.09864)：旋转位置编码。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：SwiGLU 等门控前馈结构。
- [Microsoft LoRA](https://github.com/microsoft/LoRA)：低秩增量与权重合并。
- [PyTorch 缩放点积注意力](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)：分组查询的头映射与 mask 语义。
