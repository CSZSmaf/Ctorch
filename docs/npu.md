# OpenVINO / Intel NPU

NPU 是独立的实验性推理扩展，默认关闭。模块使用 OpenVINO Runtime 编译静态 ONNX 或 IR 模型，设备显式选择为 `NPU`。它不参与 CPU/CUDA 自动求导，也不将 `.ct` 权重文件自动转换成计算图。

构建需启用 `CTORCH_USE_NPU` 并提供 OpenVINO 2026.0+ 的安装前缀。安装应用链接 `ctorch::npu`，C++ 入口为 `<dl/npu.h>`，C17 入口为 `<dl/npu_c_api.h>`。

## 模型与请求

`CompiledModel` 读取模型，按可选输入形状重塑，再编译到 NPU。输入与输出端口提供顺序、名称、dtype 和固定 shape。模型支持范围由 OpenVINO 插件、设备及驱动决定。

```cpp
dl::npu::CompiledModel model("model.onnx", {
    .performance = dl::npu::Performance::latency,
    .input_shapes = {{1, 3, 224, 224}}
});
auto request = model.create_request();
request.set_input(0, image);
request.infer();
auto prediction = request.output();
```

`image` 使用模型规定的 shape、dtype 和预处理。请求独占其输入输出缓冲，并保留编译模型的生命周期。连续推理复用同一个请求；多个请求可独立执行。

## 数据与异步执行

`set_input` 按逻辑顺序复制到请求缓冲，CUDA 输入先传回 CPU。调用结束后，原输入可释放或更新。`output()` 返回独立 CPU 张量；`output_into()` 复制到应用准备的连续、可写 CPU 张量，复用接收缓冲。

```cpp
request.set_input(0, next_image);
request.start_async();
// 可在此整理后续输入或处理另一请求的输出。
request.wait();
request.output_into(0, prediction);
```

输出在 `infer()` / `wait()` 返回，或 `ready()` 为 true 后读取。同一请求的设置、执行、读取和状态重置串行安排。`reset_state()` 为有状态模型开始新的独立序列；`optimal_request_count()` 提供运行时建议的并行请求数量。

宿主交换支持 f32、f64、i64、bool、f16 和 bf16，不表示所有类型均能作为任意 NPU 模型的端口。该接口采用宿主复制，不提供 remote tensor 或设备零拷贝。

用法见 [C++ 推理示例](../extensions/npu/examples/infer.cpp)和 [C17 推理示例](../extensions/npu/examples/infer.c)。
