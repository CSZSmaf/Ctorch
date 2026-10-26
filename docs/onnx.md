# 导出 ONNX 推理模型

ONNX 文件包含计算图、输入输出形状和固定权重。导出的模型可交给 ONNX Runtime，或由 [OpenVINO 扩展](npu.md)读取。导出模块是独立的 CMake 目标，使用时不需要安装 Python、Protobuf 或 ONNX Runtime。

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

C17 入口位于 `<dl/onnx_c_api.h>`，`dl_onnx_save()` 接收 C 图对象、输出列表和权重绑定。

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

完整两层分类器见 [export_onnx.cpp](../examples/export_onnx.cpp)。训练完成后，可以使用已加载检查点的参数构建同样的图。

## 支持范围

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
