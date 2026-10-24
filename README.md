# Ctorch

Ctorch 是面向 CPU 与单卡 CUDA 的轻量深度学习框架，以 C++20 和 C17 实现。模型使用普通 C++ 函数组织，张量、一阶自动求导、模块和优化器共享同一套计算语义。

从[在线教程](https://ctorch.one-forth.com)开始：在浏览器中观察张量布局、梯度与训练过程，再运行对应的 C++ 示例。[离线教程包](https://ctorch.one-forth.com/Ctorch-tutorial.zip)包含课程、手册、示例代码和数学字体，解压后打开 `index.html` 即可阅读。

[One Forth](https://one-forth.com) · [GitHub 仓库](https://github.com/CSZSmaf/Ctorch) · [版本下载](https://github.com/CSZSmaf/Ctorch/releases/latest)

Release 提供对应版本的源码和教程压缩包。源码包使用本地编译器构建。最新教程可通过上方的离线教程链接下载。

## 特性

- 动态形状、共享存储视图、广播、索引和一阶反向自动求导。
- Linear、分组卷积、池化、归一化、Embedding、RNN/GRU/LSTM 与 Transformer。
- SGD、Adam、AdamW、RMSprop、Adagrad，参数分组、冻结、裁剪与学习率调度。
- 批次整理、输入预取、微批累计、评估、选模和完整训练状态保存。
- CUDA、cuBLAS、可选 cuDNN，FP16/BF16 混合精度、融合注意力与激活重算。
- 显式随机 key、函数式状态、固定执行计划与自定义 VJP。
- 薄 C17 接口、可选 DLPack 交换与实验性 OpenVINO/NPU 推理扩展。

## 构建

需要 CMake 3.24+、Ninja 和支持 C++20/C17 的编译器。

```sh
cmake --preset cpu
cmake --build --preset cpu
```

预设把产物写入源码目录旁的 `Ctorch-build`。CPU 构建无需 CUDA、BLAS 或其他机器学习框架。共享库、CUDA、cuDNN 和安装方式见[构建说明](docs/build.md)。

## 使用

```cpp
#include <dl/dl.h>

auto x = dl::ones({2, 3}, {.dtype = dl::DType::f64, .requires_grad = true});
auto loss = dl::mean(x * x);
loss.backward();
auto gradient = *x.grad();
```

应用链接 CMake 目标 `ctorch::ctorch`。C++ 入口为 `<dl/dl.h>`，C17 入口为 `<dl/c_api.h>`。完整模型与训练入口见[示例](examples/README.md)。

## 文档

| 主题 | 内容 |
| --- | --- |
| [交互教程](docs/tutorial/README.md) | 十三节课程、交互演示与可编译示例 |
| [项目教程](docs/projects/README.md) | MLP、LeNet、ResNet、Transformer 与三框架实测对比 |
| [架构](docs/architecture.md) | 张量、求导、数值层与后端职责 |
| [张量与梯度](docs/tensors.md) | 类型、布局、别名、算子和梯度生命周期 |
| [训练](docs/training.md) | 模块、优化器、数据、精度和保存恢复 |
| [组合与执行](docs/execution.md) | 函数式状态、随机 key、执行计划与重算 |
| [扩展](docs/extensions.md) | C17、自定义 VJP 与 DLPack |
| [NPU 推理](docs/npu.md) | 可选 OpenVINO 请求与输出复用 |

## 独立验证

小型正确性检查位于 `tests/`，默认不参与库构建，也不随库安装。构建产物位于源码目录外：

```sh
cmake --preset cpu -DCTORCH_BUILD_TESTS=ON
cmake --build --preset cpu
ctest --test-dir ../Ctorch-build/cpu --output-on-failure
```

覆盖范围与外部验证入口见[测试说明](tests/README.md)。Linux 与 Windows 的 CPU 工作流复用这套入口。

框架面向单机稠密训练。一阶求导、显式设备与精度选择构成基础接口；分布式训练、高阶求导、稀疏训练和通用 JIT 不在当前接口范围内。
