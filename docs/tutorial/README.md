# Ctorch 教程

访问[在线教程](https://ctorch.one-forth.com)，或下载[离线教程包](https://ctorch.one-forth.com/Ctorch-tutorial.zip)。离线包解压后打开根目录的 `index.html`。课程、手册、示例代码和数学字体均包含在压缩包中，阅读时无需安装依赖。

[One Forth](https://one-forth.com) · [GitHub 仓库](https://github.com/CSZSmaf/Ctorch)

基础课程按“快速开始 → 张量 → 数据与预处理 → 模型 → 自动求导 → 训练与评估 → 保存恢复”的顺序组织。后续课程介绍自定义模块、设备与混合精度、函数式状态与执行计划、扩展接口。最后一节提供 API、示例和常见问题索引。

保存恢复之后是[实战项目](../projects/README.md)，包括 MNIST MLP、LeNet、CIFAR-10 ResNet-20 和 Shakespeare Transformer。[框架结果对比](../projects/comparison.md)提供三框架实际训练曲线、数值误差和耗时，曲线支持逐步查看批次损失。

八个交互演示说明训练步骤、张量布局、分类概率与损失、梯度累计、SGD、参数绑定、状态快照和算子融合。演示由 JavaScript 计算；C++ 示例需要使用 Ctorch 编译运行。SGD 演示与 [first_training.cpp](../../examples/first_training.cpp) 使用相同样本和更新公式，浏览器使用双精度，C++ 示例使用 FP32。张量、数据和求导章节还提供可以展开答案的练习。

[training_basics.cpp](../../examples/training_basics.cpp) 包含批次分类、验证指标、学习率调度、保存恢复和新样本预测。[parameter_layer.cpp](../../examples/parameter_layer.cpp) 介绍自定义参数层。其余完整程序见[示例目录](../../examples/README.md)。

左侧目录切换章节，右侧目录定位正文。`Ctrl K` 或 `⌘ K` 搜索课程正文和代码。页面支持代码复制、主题切换、已读标记和打印。[手册](../README.md)和[示例](../../examples/README.md)在网站内阅读，源文件页面提供下载按钮。

课程顺序参考 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。数据预处理、分类输出和预测步骤参考 [TensorFlow 分类教程](https://www.tensorflow.org/tutorials/keras/classification)，训练与评估的组织参考 [TensorFlow 自定义训练循环](https://www.tensorflow.org/tutorials/quickstart/advanced)。代码使用 Ctorch 自身的 C++ 接口。

## 生成网站文件

使用 Node.js 22 或更高版本。在仓库根目录运行：

```sh
npm install --prefix ../Ctorch-docs-tools --save-exact katex@0.19.1 marked@18.1.1
node docs/tutorial/build.mjs ../Ctorch-site ../Ctorch-docs-tools
```

第一个参数指定输出目录，第二个参数指定生成工具的安装目录。生成后打开 `../Ctorch-site/index.html`，也可以把整个输出目录部署到静态网站。

手册由仓库中的 Markdown 生成。公式使用 [KaTeX](https://katex.org/) 在生成时排版，字体与许可文件一并复制到输出目录。网页不依赖第三方 CDN。在线网站与离线包使用同一组相对链接。

直接打开仓库中的 `docs/tutorial/index.html` 可以阅读课程和使用交互演示；数学排版及手册页面需要使用上述生成结果。
