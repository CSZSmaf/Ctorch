# Ctorch 教程

访问[在线教程](https://ctorch.one-forth.com)，或从 [Release](https://github.com/CSZSmaf/Ctorch/releases/latest) 下载离线教程包。离线包解压后打开根目录的 `index.html`。下载整个仓库时，打开 `docs/tutorial/index.html`。网页可离线使用，无需安装前端依赖。

[One Forth](https://one-forth.com) · [GitHub 仓库](https://github.com/CSZSmaf/Ctorch)

基础课程按“快速开始 → 张量 → 数据与预处理 → 模型 → 自动求导 → 训练与评估 → 保存恢复”的顺序组织。后续课程介绍自定义模块、设备与混合精度、函数式状态与执行计划、扩展接口。最后一节提供 API、示例和常见问题索引。

八个交互演示说明训练步骤、张量布局、分类概率与损失、梯度累计、SGD、参数绑定、状态快照和算子融合。演示由 JavaScript 计算；C++ 示例需要使用 Ctorch 编译运行。SGD 演示与 [first_training.cpp](../../examples/first_training.cpp) 使用相同样本和更新公式，浏览器使用双精度，C++ 示例使用 FP32。张量、数据和求导章节还提供可以展开答案的练习。

[training_basics.cpp](../../examples/training_basics.cpp) 包含批次分类、验证指标、学习率调度、保存恢复和新样本预测。[parameter_layer.cpp](../../examples/parameter_layer.cpp) 介绍自定义参数层。其余完整程序见[示例目录](../../examples/README.md)。

左侧目录切换章节，右侧目录定位正文。`Ctrl K` 或 `⌘ K` 搜索正文和代码。页面支持代码复制、主题切换、已读标记和打印。

课程顺序参考 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。数据预处理、分类输出和预测步骤参考 [TensorFlow 分类教程](https://www.tensorflow.org/tutorials/keras/classification)，训练与评估的组织参考 [TensorFlow 自定义训练循环](https://www.tensorflow.org/tutorials/quickstart/advanced)。代码使用 Ctorch 自身的 C++ 接口。

## 生成网站文件

`node docs/tutorial/build.mjs ../Ctorch-site` 将网页写入指定目录。在线页面的手册和源码链接指向 GitHub，仓库中的相对链接继续用于离线阅读。
