# 独立正确性检查

此目录保存规模小、可重复的核心检查，使用 C++20、C17 和 CTest，无额外测试框架依赖。库与示例分别位于 `src/` 和 `examples/`；测试默认关闭，不参与库安装。

```sh
cmake -S . -B ../Ctorch-build/check -DCTORCH_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build ../Ctorch-build/check --config Debug
ctest --test-dir ../Ctorch-build/check -C Debug --output-on-failure
```

在仓库根目录执行。测试程序、临时检查点、安装包及消费工程都写入指定的构建目录。

| 程序 | 覆盖内容 |
| --- | --- |
| tensor | 共享视图、宿主类型转换、广播与梯度累计，eager 与融合/非融合计划的一致性 |
| gradients | FP32 常量与近常量统计、归一化状态和梯度；FP64 方向中心有限差分，涵盖激活、广播、矩阵乘、视图、归一化、卷积、池化、注意力、重复索引和重算 |
| state | 共享参数、稳定命名、登记约束、快照、旧别名恢复、函数式绑定与梯度还原 |
| c_api | C17 创建、前向、梯度、拥有权释放与错误边界 |
| installed_package | 安装后由独立 C/C++ 工程通过 `find_package` 编译并运行 |

有限差分检查避开不可微分点，测试失败会返回非零退出码。CPU 工作流在 Linux Debug 静态库和 Windows Release 共享库上运行同一入口。

GPU/NPU 测试、长时间训练和性能比较可放在独立开发目录。指定 `CTORCH_DEVELOPMENT_DIR` 后，CMake 使用其中的 `tests/` 替代本目录；`CTORCH_BUILD_BENCHMARKS` 启用其中的 `benchmarks/`。
