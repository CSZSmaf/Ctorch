# Ctorch 教程

下载仓库后，在浏览器打开 [index.html](index.html)。网页可离线使用，无需安装前端依赖。GitHub 文件预览展示 HTML 源码。

十二节课程覆盖快速开始、张量、自动求导、数据加载、常用模型、优化器与训练、自定义模块、保存恢复、混合精度、函数式状态与执行计划、扩展接口，以及 API 和示例索引。

七个交互演示说明训练步骤、张量布局、梯度累计、SGD、参数绑定、状态快照和算子融合。演示由 JavaScript 计算；C++ 示例需要使用 Ctorch 编译运行。SGD 演示与 [first_training.cpp](../../examples/first_training.cpp) 使用相同样本和更新公式，浏览器使用双精度，C++ 示例使用 FP32。

[training_basics.cpp](../../examples/training_basics.cpp) 包含批次分类、验证指标、学习率调度和保存恢复。[parameter_layer.cpp](../../examples/parameter_layer.cpp) 介绍自定义参数层。其余完整程序见[示例目录](../../examples/README.md)。

左侧目录切换章节，右侧目录定位正文。`Ctrl K` 或 `⌘ K` 搜索正文和代码。页面支持代码复制、主题切换、已读标记和打印。

课程和导航结构参考 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 与 [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)。
