# 构建与安装

CPU 核心使用 C++20、C17 和 CMake 3.24+。预设采用 Ninja；也可通过 `cmake -S . -B <build-directory>` 选择其他生成器。构建选项不会自动下载依赖。

## CPU 与共享库

```sh
cmake --preset cpu
cmake --build --preset cpu

cmake --preset shared
cmake --build --preset shared
```

产物分别位于相邻的 `Ctorch-build/cpu` 与 `Ctorch-build/shared`。`CMAKE_BUILD_TYPE` 默认是 Release，多配置生成器使用 `--config Release`。

## 可选后端

| 选项 | 依赖与作用 |
| --- | --- |
| CTORCH_USE_BLAS | 具有 CBLAS 接口的 BLAS，使用 32 位整数维度 |
| CTORCH_USE_CUDA | CUDA Toolkit 12+ 及其支持的 C++20 宿主编译器 |
| CTORCH_USE_CUDNN | CUDA、cuDNN 9.7+、cuDNN frontend 1.12+ |
| CTORCH_USE_DLPACK | 官方 DLPack 1.x 头文件 |
| CTORCH_USE_NPU | OpenVINO 2026.0+ Runtime，构建独立推理扩展 |
| CTORCH_BUILD_EXAMPLES | 构建 C/C++ 接口与训练示例 |

这些选项默认关闭。CUDA 构建使用 `cuda` 或 `cudnn` 预设，GPU 架构通过 `CMAKE_CUDA_ARCHITECTURES` 指定。

```sh
cmake --preset cuda
cmake --build --preset cuda

cmake --preset cudnn
cmake --build --preset cudnn
```

依赖不在默认搜索目录时，使用 `CMAKE_PREFIX_PATH`、`CUDAToolkit_ROOT`、`CUDNN_ROOT`、`CUDNNFrontend_ROOT` 或 `DLPack_ROOT` 指定其安装前缀。`DLPack_ROOT` 下应包含 `include/dlpack/dlpack.h`。

CBLAS 可通过 `BLA_VENDOR` 选择实现，通过 `CTORCH_CBLAS_INCLUDE_DIR` 指定头文件目录。OpenBLAS 的 `include/openblas` 布局会自动参与查找。

## 安装与调用

```sh
cmake --install ../Ctorch-build/cpu --prefix ../Ctorch-install
```

应用工程使用安装包提供的目标：

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyModel LANGUAGES C CXX)
find_package(Ctorch 0.1 CONFIG REQUIRED)
add_executable(model main.cpp)
target_link_libraries(model PRIVATE ctorch::ctorch)
```

配置应用时，将安装前缀加入 `CMAKE_PREFIX_PATH`。C++ 应用与库使用兼容的编译器、标准库和运行库；C17 文件由 C 编译器处理，最终使用 C++ 链接器。

Windows 共享库调用需要将安装前缀的 `bin` 及所选后端的运行库目录加入当前进程的 DLL 搜索路径。安装包不复制 CUDA、cuDNN、BLAS 或 OpenVINO 的运行库。

`examples/installed` 演示仅通过安装包取得库：

```sh
cmake -S examples/installed -B ../Ctorch-build/installed-examples \
  -DCMAKE_PREFIX_PATH=../Ctorch-install
cmake --build ../Ctorch-build/installed-examples
```

可选 NPU 目标为 `ctorch::npu`，通过 `find_package(Ctorch CONFIG REQUIRED COMPONENTS npu)` 请求。其他安装目标继续使用 `ctorch::ctorch`。
