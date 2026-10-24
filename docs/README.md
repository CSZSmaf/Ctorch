# Ctorch 手册

[交互教程](tutorial/index.html)按顺序介绍张量、自动求导、训练和扩展。以下手册说明接口约定与构建方法。

| 手册 | 内容 |
| --- | --- |
| [项目教程](projects/README.md) | 真实数据训练、经典模型和中等规模教学项目 |
| [框架结果对比](projects/comparison.md) | 相同初值训练、输出与梯度误差、CPU 耗时 |
| [构建与安装](build.md) | CPU、CUDA、cuDNN、OpenVINO 和 CMake 安装包 |
| [张量与梯度](tensors.md) | 数据类型、设备、视图、算子与反向传播 |
| [模型与训练](training.md) | 模块、优化器、数据加载、混合精度和检查点 |
| [组合与执行](execution.md) | 函数式状态、随机 key、执行计划和激活重算 |
| [C17 与外部计算](extensions.md) | C 接口、自定义 VJP、DLPack 和 CUDA 交接 |
| [OpenVINO / Intel NPU](npu.md) | 编译推理模型、输入输出与异步请求 |
| [架构](architecture.md) | 模块分层、存储拥有权和后端组织 |

## 示例与测试

[示例目录](../examples/README.md)列出可编译的训练和接口示例。源文件页面提供完整代码和下载入口。

[测试说明](../tests/README.md)介绍独立检查的构建和运行方法。

## 教程使用

[教程说明](tutorial/README.md)介绍课程内容、交互操作和网站生成方法。
