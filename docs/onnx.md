# 导出 ONNX 推理模型

ONNX 文件保存推理所需的计算图和权重。分类器和卷积网络可以用 `Graph` 描述后导出。Ctorch 的 `TransformerDecoder` 可以直接从模型导出，支持不同长度的输入和 KV 缓存。导出器是独立的 CMake 目标，只依赖 Ctorch。运行文件时，推理引擎需要支持模型使用的 ONNX 算子。

## 构建

```sh
cmake -S . -B ../Ctorch-build/onnx -G Ninja -DCMAKE_BUILD_TYPE=Release -DCTORCH_USE_ONNX=ON -DCTORCH_BUILD_EXAMPLES=ON
cmake --build ../Ctorch-build/onnx
../Ctorch-build/onnx/bin/ctorch_export_onnx ../model.onnx
```

应用包含 `<dl/onnx.h>`，链接 `ctorch::onnx`。安装后的应用使用：

```cmake
find_package(Ctorch CONFIG REQUIRED COMPONENTS onnx)
target_link_libraries(model PRIVATE ctorch::onnx)
```

C17 入口位于 `<dl/onnx_c_api.h>`。`dl_onnx_save` 接收图、输出列表和权重绑定。`dl_onnx_save_decoder` 接收解码器及可选的词嵌入和输出层。

## 构图和权重绑定

模型使用[固定执行计划](execution.md)中的 `Graph` 描述。普通输入和模型权重都先注册为图输入，导出时指定哪些输入固定为权重。

```cpp
dl::nn::Linear layer(4, 3);
dl::plan::Graph graph;
auto x = graph.input(dl::contiguous_spec({2, 4}), "input");
auto w = graph.input(dl::tensor_spec(layer.weight()), "weight");
auto b = graph.input(dl::tensor_spec(*layer.bias()), "bias");
auto y = graph.call(dl::plan::ops::linear(), {x, w, b});

dl::onnx::save("model.onnx", graph, {y}, {
    {w, layer.weight()}, {b, *layer.bias()},
});
```

这个文件保留名为 `input` 的 `[2,4]` 输入，输出名为 `output_0`，形状为 `[2,3]`。权重作为常量写入文件。多个输出依次使用 `output_1`、`output_2` 等名称，这些名称由导出器保留。

完整两层分类器见 [export_onnx.cpp](../examples/export_onnx.cpp)。

## 部署训练好的 MNIST 分类器

[MNIST 项目](projects/mnist.md)训练 MLP 后会保存 `best.ct`。把这份权重作为第二个参数交给导出程序。最后一个参数指定每次推理的图像数，默认是 1。

```sh
../Ctorch-build/onnx/bin/ctorch_export_onnx ../mnist.onnx ../Ctorch-work/runs/mnist-mlp/best.ct 1
```

下面用 ONNX Runtime 识别测试集的第一张图像。MNIST 图像文件的前 16 字节是文件头。图像按行存储，读取后除以 255，得到与训练相同的输入。

```python
from pathlib import Path
import numpy as np
import onnxruntime as ort

images = Path("../Ctorch-work/data/mnist/t10k-images-idx3-ubyte")
pixels = np.fromfile(images, dtype=np.uint8, count=28 * 28, offset=16)
image = pixels.reshape(1, 1, 28, 28).astype(np.float32) / 255
session = ort.InferenceSession("../mnist.onnx", providers=["CPUExecutionProvider"])
scores = session.run(["output_0"], {"input": image})[0]
print(scores.argmax(axis=1).item())
```

输出的十列依次对应数字 0 到 9。分数最大的列给出预测类别。部署时只需复制 `mnist.onnx`，模型权重已经包含在文件中。

## 导出带 KV 缓存的解码器

本节示例使用 [Ctorch 主分支源码](https://github.com/CSZSmaf/Ctorch/tree/main)。把训练时使用的词嵌入、解码器和输出层交给 `save_decoder`。导出器读取模型的层数、头数、旋转位置表和权重。输入是 `int64` 的 `input_ids[batch,tokens]`，输出是 `logits[batch,tokens,输出维度]`。词嵌入和输出层可以共享权重，也可以各自训练。

```cpp
dl::onnx::save_decoder("language_model.onnx", embedding, decoder, output_layer);
```

只需要解码器特征时，调用 `dl::onnx::save_decoder("decoder.onnx", decoder)`。这个模型接收 `features[batch,tokens,width]` 并返回同形状的 `hidden_states`，两者都是 FP32。两个接口都导出 FP32 权重和 ONNX opset 24 的标准算子。[ONNX Attention](https://onnx.ai/onnx/operators/onnx__Attention.html)

例如，[字节解码器项目](projects/decoder.md)训练后，用下面的命令导出可以直接续写的语言模型。

```sh
../Ctorch-build/onnx/bin/ctorch_export_decoder_onnx ../Ctorch-work/runs/decoder/best.ct ../decoder.onnx
```

每层有两份缓存。第 `i` 层接收 `past_key_i` 和 `past_value_i`，返回 `present_key_i` 和 `present_value_i`。形状依次为 `[batch,kv_heads,历史长度,head_dim]` 和 `[batch,kv_heads,历史长度+本次输入长度,head_dim]`。首次输入时把历史长度设为零。续写时把上次返回的缓存交给同一层。累计长度可以达到模型的 RoPE 表容量。

下面用 ONNX Runtime 运行字节模型。第一次送入完整提示文本，第二次只送入新选出的字节。

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("../decoder.onnx", providers=["CPUExecutionProvider"])
prefix = np.array([list(b"ROMEO: ")], dtype=np.int64)
cache = {
    port.name: np.empty((len(prefix), port.shape[1], 0, port.shape[3]), dtype=np.float32)
    for port in session.get_inputs() if port.name.startswith("past_")
}

def decode(ids, cache):
    values = session.run(None, {"input_ids": ids, **cache})
    outputs = dict(zip((port.name for port in session.get_outputs()), values))
    cache = {
        name.replace("present_", "past_", 1): value
        for name, value in outputs.items() if name.startswith("present_")
    }
    return outputs["logits"], cache

logits, cache = decode(prefix, cache)
next_id = logits[:, -1].argmax(axis=-1, keepdims=True).astype(np.int64)
logits, cache = decode(next_id, cache)
```

批量和每次输入的 token 数由调用方决定。同一批序列共用一个历史长度。分词和选取下一个 token 由应用完成。[导出示例源码](../examples/export_decoder_onnx.cpp)

## 固定形状图支持的算子

导出格式为 ONNX opset 18、IR 8。输入形状固定，支持 FP32 和 FP64 图。权重的形状和类型与对应图输入一致；CPU 和 CUDA 权重均可导出，CUDA 权重在导出时复制到 CPU。

| 计算 | 图描述 |
| --- | --- |
| 广播四则运算 | `add`、`sub`、`mul`、`div` |
| 激活 | `relu`、`add_relu`、`sigmoid`、`tanh`、`silu`、`softmax` |
| 矩阵与线性层 | `matmul`、`linear` |
| 布局 | `reshape`、`transpose`、`contiguous` |
| 一维和二维空间运算 | 带批维的分组卷积、最大池化、平均池化 |
| 全局平均池化 | 输出尺寸为 1 的 `adaptive_avg_pool1d/2d` |
| 归一化 | LayerNorm；使用运行统计的评估模式 BatchNorm |
| Dropout | 评估模式，或概率为零时的恒等计算 |

Linear、SiLU 和 LayerNorm 等操作可以展开为标准 ONNX 节点，部署端按其优化器选择执行方式。模型使用哪些算子和精度，也决定了推理引擎可使用哪些设备。

`Graph` 是显式描述，不自动分析任意 C++ 函数中的分支。自定义回调、训练随机操作和状态更新需要先转换成支持的推理计算；导出器会给出不支持的节点名称。未参与输出的纯计算会被移除。

## 用 ONNX Runtime 读取

下面的 Python 代码只用于部署应用或结果检查，依赖与 Ctorch 主库分开安装。

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
x = np.ones((2, 4), dtype=np.float32)
y = session.run(None, {"input": x})[0]
```

参考：[ONNX 模型与计算图](https://onnx.ai/onnx/intro/concepts.html)、[ONNX Runtime Python 接口](https://onnxruntime.ai/docs/api/python/api_summary.html)。
