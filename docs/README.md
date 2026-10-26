# Ctorch 手册

[交互教程](tutorial/index.html)按顺序介绍张量、自动求导、训练和扩展。以下手册说明接口约定与构建方法。

| 手册 | 内容 |
| --- | --- |
| [构建与安装](build.md) | CPU、CUDA、cuDNN、OpenVINO 和 CMake 安装包 |
| [张量与梯度](tensors.md) | 数据类型、设备、视图、算子与反向传播 |
| [从 PyTorch 迁移](migration.md) | 接口对应、参数绑定、注意力、状态与数值对齐 |
| [模型与训练](training.md) | 模块、优化器、数据加载、混合精度和检查点 |
| [数据加载](data.md) | 多输入批次、变长序列、后台预取和采样 |
| [损失函数](objectives.md) | 类别权重、标签平滑、分割损失和 KL 散度 |
| [训练控制](training-tools.md) | 早停、平台期调度、EMA 和评价指标 |
| [组合与执行](execution.md) | 函数式状态、随机 key、执行计划和激活重算 |
| [性能与内存](performance.md) | 缓冲复用、C++ 内层循环和模型计时 |
| [C17 与外部计算](extensions.md) | C 接口、自定义 VJP、DLPack 和 CUDA 交接 |
| [OpenVINO / Intel NPU](npu.md) | 编译推理模型、输入输出与异步请求 |
| [ONNX 导出](onnx.md) | 固定形状推理图、权重绑定和部署 |
| [架构](architecture.md) | 模块分层、存储拥有权和后端组织 |

## 示例与测试

[示例目录](../examples/README.md)列出可编译的训练和接口示例。源文件页面提供完整代码和下载入口。

[测试说明](../tests/README.md)介绍独立检查的构建和运行方法。

## 教程使用

[教程说明](tutorial/README.md)介绍课程内容、交互操作和网站生成方法。
