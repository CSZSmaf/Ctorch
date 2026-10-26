# 从 PyTorch 迁移

Ctorch 的张量运算、模块和训练循环使用 C++ 接口。迁移模型时，先对齐形状、参数、设备、精度和训练模式，再比较输出、梯度及参数更新。

## 常用接口与行为

| PyTorch 用法 | Ctorch 用法与行为 |
| --- | --- |
| `tensor.to(device)` | `tensor.to(dl::Device::cuda())`；类型转换单独调用 `astype()` |
| `tensor.item()` | `tensor.item<float>()` 等；读取时转换为指定的宿主数值类型 |
| `tensor.detach()` | `tensor.detach()` 共享数据并结束求导历史；独立数据使用 `clone()` |
| `loss.backward()` | `loss.backward()`；叶梯度累计，一次反向消费该前向图 |
| `optimizer.zero_grad()` | 同名调用清空梯度槽；下一次反向重新写入 |
| `model.train()` / `model.eval()` | 同名调用设置模块模式；评估计算配合 `dl::NoGradGuard` |
| `state_dict()` | 状态表共享当前存储；独立快照使用 `dl::clone_state()` |
| `torch.autocast` | `dl::AutocastGuard`；低精度算子的激活输出采用 FP32 |
| `GradScaler.step()` 和 `update()` | `scaler.step(optimizer)` 一次完成梯度还原、条件更新与倍率调整 |
| `torch.compile(model)` | `dl::plan::Graph` 显式建立固定计算图，再调用 `compile()` |

保存的张量在反向前保持原值。设备迁移、状态加载和参数更新安排在前向图之间。先迁移模型，再用迁移后的参数构造优化器。

## 自定义参数层

参数在构造时登记，前向通过 `bound()` 获取当前调用使用的参数。这样普通调用和函数式调用共用同一段计算。

```cpp
class Scale : public dl::nn::UnaryModule {
public:
    Scale() : weight_(register_parameter(
        "weight", dl::full({}, 2.0, {.requires_grad = true}))) {}

    dl::Tensor forward(const dl::Tensor& x) override {
        return x * bound(weight_);
    }

private:
    dl::Tensor weight_;
};
```

普通调用的 `bound(weight_)` 返回模型自身的参数。`FunctionalModule` 调用则返回这次绑定的外部参数。buffer 使用 `register_buffer()` 和 `bound_buffer()`。完整的普通训练、外部状态调用和梯度更新见[自定义参数层示例](../examples/parameter_layer.cpp)。

## 注意力和编码器

`MultiHeadAttention` 和编码器的输入为 `[batch, length, dimensions]`。底层缩放点积注意力使用带头维的布局。布尔 `keep_mask` 中 `true` 表示保留该位置，每一行至少保留一个位置。PyTorch 各注意力接口的布尔 mask 含义不同，迁移时按实际使用的接口转换。

`TransformerEncoderLayer` 采用 PreNorm：先归一化，再计算注意力或前馈子层，最后加残差。前馈激活为 GELU。`TransformerEncoder` 在全部编码层之后执行一次 LayerNorm。

用 PyTorch 对齐这组结构时，设置 `batch_first=True`、`norm_first=True`、`activation="gelu"`，并为编码器提供最终 LayerNorm。比较推理时关闭 dropout；比较训练时固定输入、参数和随机掩码。对应选项见 [PyTorch 编码层文档](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoderLayer.html)。

## 保存模型与继续训练

```cpp
auto best = dl::clone_state(model.state_dict());
// 后续训练可以继续更新 model，best 保留采集时的数值。
model.load_state_dict(best);
```

只加载权重适用于推理和重新开始训练。继续原有训练还需要恢复优化器、调度器、随机流、数据游标以及混合精度状态。训练快照在参数更新边界采集，具体用法见[模型与训练](training.md)。

## 对齐数值和效率

先用相同参数和输入比较前向，再用相同上游梯度比较反向，最后比较一步参数更新。BatchNorm 还需要比较运行均值和运行方差，AdamW 需要比较一阶矩、二阶矩及步数。

低精度输入先按相同类型量化。Ctorch 的低精度激活输出为 FP32，而 PyTorch 的部分算子保留 FP16/BF16 输出，两者的后续舍入位置可能不同。[张量与梯度](tensors.md)列出各类运算的精度约定。

完整训练比较包含清理梯度、前向、损失、反向和更新。CUDA 在计时边界同步；首次编译与稳定运行分别计时。可参考[跨框架结果](projects/comparison.md)和[多维性能比较](projects/performance.md)的配置。
