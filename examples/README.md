# 示例

使用 `CTORCH_BUILD_EXAMPLES=ON` 构建示例。接口示例在 `targets.cmake` 登记，源码与独立安装工程共用这份清单。示例构建不会下载数据或自动开始训练。

```sh
cmake --preset cpu -DCTORCH_BUILD_EXAMPLES=ON
cmake --build --preset cpu
```

## 真实训练

| 程序 | 输入 | 模型 |
| --- | --- | --- |
| ctorch_mnist_mlp | 解压后的 MNIST IDX 文件 | 两层 MLP |
| ctorch_mnist_cnn | 相同的 MNIST 文件 | 卷积、BatchNorm、池化与分类器 |
| ctorch_mnist_lenet | 相同的 MNIST 文件 | LeNet 风格卷积分类器 |
| ctorch_cifar_resnet | CIFAR-10 binary 目录 | 272,474 参数的 ResNet-20 |
| ctorch_byte_transformer | 本地文本文件 | 字节/位置 Embedding、因果 Transformer、共享词表权重 |
| ctorch_shakespeare_transformer | 本地文本文件 | 4 层、宽度 128、上下文 128 的 Transformer |
| ctorch_text_generate | 文本模型权重与前缀 | 自回归字节采样 |

```text
ctorch_mnist_mlp <mnist-directory> <output-directory> [--device cpu|index] [--epochs N] [--batch-size N]
ctorch_mnist_cnn <mnist-directory> <output-directory> [--device cpu|index] [--epochs N] [--batch-size N]
ctorch_byte_transformer <text-file> <output-directory> [--device cpu|index] [--steps N] [--batch-size N] [--eval-every N]
```

训练支持 `--precision f32|bf16|f16`，默认 f32；文本模型另接受 `--attention math|cudnn`，默认 math。cuDNN 注意力需要对应后端、CUDA 设备及 f16/bf16 精度。`--stop-after N` 控制更新次数，`--resume <checkpoint>` 用于从训练状态恢复，具体参数由各入口的 usage 说明给出。

MNIST 目录包含标准解压文件名，像素缩放到 `[0,1]`。训练集前 55,000 个样本用于更新，后 5,000 个用于验证，官方测试集用于最终评估。

字节模型按原始文件的 90%/5%/5% 划分训练、验证和测试区间，再构造序列。目标是下一个字节，不足长度的目标使用忽略标签。程序共用加权指标、评估、最佳模型选择和更新边界快照；数据与输出位置由参数指定。

## 接口示例

项目的下载、训练、恢复和预测步骤见[项目教程](../docs/projects/README.md)。[比较示例](comparison/README.md)使用相同数据和权重对比 PyTorch、TensorFlow 的结果，Python 依赖单独安装。

| 文件 | 主题 |
| --- | --- |
| [first_training.cpp](first_training.cpp) | 从零初值学习线性回归，对应交互教程 |
| [training_basics.cpp](training_basics.cpp) | 批次分类、验证指标、学习率调度、保存恢复与新样本预测；可传入检查点文件名 |
| [parameter_layer.cpp](parameter_layer.cpp) | 带参数的自定义层、普通与函数式绑定 |
| [custom_layer.cpp](custom_layer.cpp) / [c_train.c](c_train.c) | C++ 自定义层与 C17 训练 |
| [finetune.cpp](finetune.cpp) / [recurrent_train.cpp](recurrent_train.cpp) | 分阶段微调与 GRU 训练 |
| [gradients.cpp](gradients.cpp) / [c_gradient_processing.c](c_gradient_processing.c) | 显式梯度、范数与裁剪 |
| [functional_module.cpp](functional_module.cpp) / [functional_optimizer.cpp](functional_optimizer.cpp) | 参数与状态交接 |
| [functional_state.cpp](functional_state.cpp) / [random_key.cpp](random_key.cpp) | BatchNorm 状态与显式随机 key |
| [execution_plan.cpp](execution_plan.cpp) / [multi_output_plan.cpp](multi_output_plan.cpp) | 固定计划、融合和多输出 |
| [custom_vjp.cpp](custom_vjp.cpp) / [activation_checkpoint.cpp](activation_checkpoint.cpp) | 外部 VJP 与激活重算 |
| [cuda_memory.cpp](cuda_memory.cpp) | CUDA 内存池使用与回收 |
| [cudnn_attention.cpp](cudnn_attention.cpp) | BF16 因果 Transformer 与 cuDNN SDPA |
| [dlpack_vjp.cpp](dlpack_vjp.cpp) / [c_dlpack.c](c_dlpack.c) | DLPack 共享视图与外部计算 |

源码目标使用 `ctorch_` 前缀；`installed` 工程使用相同示例文件，只通过安装包链接。CUDA、cuDNN 和 DLPack 专用示例随对应构建选项启用。
