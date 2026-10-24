# 从基础开始

用九节课程学习 Ctorch。下载仓库后，在浏览器直接打开 [index.html](index.html)；网页无需网络、构建步骤或第三方前端依赖。GitHub 的文件预览只展示 HTML 源码，本地浏览器提供完整交互。

课程按“快速开始 → 张量 → 自动求导 → 训练 → 自定义层 → 状态 → 精度 → 执行 → 扩展”组织。左侧切换课程，右侧定位本节内容，`Ctrl K` 或 `⌘ K` 搜索正文与代码。界面支持浅色与深色主题、窄屏阅读、键盘导航和本节打印；完成标记保存在当前浏览器。

七个交互实验分别展示训练流程、张量布局、梯度累计、SGD、参数绑定、状态快照和局部融合。它们用 JavaScript 演示数学或状态关系，不调用本地 Ctorch，也不代表硬件性能测试。SGD 演示与 [first_training.cpp](../../examples/first_training.cpp) 使用相同的样本、零初值、MSE 和更新公式；浏览器为双精度计算，C++ 示例默认 FP32。

代码块可以复制，完整示例可以下载。带参数自定义层见 [parameter_layer.cpp](../../examples/parameter_layer.cpp)，其他训练与接口程序见[示例目录](../../examples/README.md)。

## 维护

- `index.html` 保存完整正文、代码片段与可访问的界面结构。
- `style.css` 集中管理阅读布局、主题与响应式样式。
- `tutorial.js` 管理课程导航、搜索、进度及小型数学实验。

增加课程时，在正文添加 `.lesson` 和左侧课程链接；页内目录与搜索索引从正文生成。教学代码与公开接口保持一致，构建和数值验证通过独立的 `examples/` 与 `tests/` 进行。网页调试截图和浏览器日志存放在仓库外。

## 参考

阅读结构参考 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) 与 [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) 的分步课程、侧边导航、页内目录和代码示例组织。文字、样式与交互实现围绕 Ctorch 的 C++ 接口独立编写，不依赖 PyTorch 的站点资源。
