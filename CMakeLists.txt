cmake_minimum_required(VERSION 3.24)
project(Ctorch VERSION 0.2.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_SHARED_LIBS "Build Ctorch as a shared library" OFF)
option(CTORCH_USE_BLAS "Use an installed BLAS library with a CBLAS interface" OFF)
option(CTORCH_USE_CUDA "Build CUDA float32 kernels and the cuBLAS backend" OFF)
option(CTORCH_USE_CUDNN "Use installed cuDNN and its C++ frontend for CUDA convolution and attention" OFF)
option(CTORCH_USE_NPU "Build the optional Intel NPU / OpenVINO inference extension" OFF)
option(CTORCH_USE_DLPACK "Build zero-copy tensor exchange using installed DLPack headers" OFF)
option(CTORCH_BUILD_TESTS "Build the independent numerical verification programs" OFF)
option(CTORCH_BUILD_EXAMPLES "Build the training examples" OFF)
option(CTORCH_BUILD_BENCHMARKS "Build the independent performance measurement program" OFF)
set(CTORCH_DEVELOPMENT_DIR "" CACHE PATH
    "Directory containing optional external tests and benchmarks")

add_library(ctorch
    src/core/layout.cpp
    src/core/storage.cpp
    src/core/buffer_pool.cpp
    src/core/cuda_memory.cpp
    src/core/transfer.cpp
    src/core/amp.cpp
    src/core/tensor.cpp
    src/core/spec.cpp
    src/core/inspect.cpp
    src/core/factory.cpp
    src/core/random.cpp
    src/autograd/engine.cpp
    src/autograd/grad.cpp
    src/autograd/custom.cpp
    src/autograd/checkpoint.cpp
    src/plan/graph.cpp
    src/plan/operations.cpp
    src/plan/functional.cpp
    src/plan/spatial.cpp
    src/plan/optimizers.cpp
    src/ops/numeric.cpp
    src/ops/views.cpp
    src/ops/partition.cpp
    src/ops/elementwise.cpp
    src/ops/reduction.cpp
    src/ops/statistics.cpp
    src/ops/matmul.cpp
    src/ops/indexing.cpp
    src/ops/concatenation.cpp
    src/ops/loss.cpp
    src/ops/spatial.cpp
    src/ops/resampling.cpp
    src/ops/convolution.cpp
    src/ops/normalization.cpp
    src/ops/recurrent.cpp
    src/ops/attention.cpp
    src/ops/cudnn_attention.cpp
    src/kernels/cpu/vector.c
    src/kernels/cpu/pointwise.cpp
    src/kernels/cpu/random.cpp
    src/kernels/cpu/comparison.cpp
    src/kernels/cpu/reduction.cpp
    src/kernels/cpu/normalization.cpp
    src/kernels/cpu/batch_normalization.cpp
    src/kernels/cpu/matmul.cpp
    src/kernels/cpu/indexing.cpp
    src/kernels/cpu/loss.cpp
    src/kernels/cpu/optimizer.cpp
    src/kernels/cpu/gradients.cpp
    src/kernels/cpu/spatial.cpp
    src/kernels/cpu/resampling.cpp
    src/backends/cpu.cpp
    src/nn/module.cpp
    src/nn/functional.cpp
    src/nn/layers.cpp
    src/nn/spatial.cpp
    src/nn/recurrent.cpp
    src/nn/attention.cpp
    src/optim/optimizer.cpp
    src/optim/update.cpp
    src/optim/functional.cpp
    src/optim/clipping.cpp
    src/optim/gradients.cpp
    src/optim/scaler.cpp
    src/optim/scheduler.cpp
    src/optim/state.cpp
    src/data/loader.cpp
    src/data/prefetch.cpp
    src/io/checkpoint.cpp
    src/io/training.cpp
    src/io/training_state.cpp
    src/io/optimizer.cpp
    src/io/scheduler.cpp
    src/api/common.cpp
    src/api/cuda.cpp
    src/api/tensor.cpp
    src/api/autograd.cpp
    src/api/extension.cpp
    src/api/plan.cpp
    src/api/amp.cpp
    src/api/transfer.cpp
    src/api/statistics.cpp
    src/api/spatial.cpp
    src/api/recurrent.cpp
    src/api/optimizer.cpp
    src/api/scheduler.cpp
    src/api/module.cpp
    src/api/functional.cpp
    src/api/state.cpp
)
add_library(ctorch::ctorch ALIAS ctorch)

target_compile_features(ctorch PUBLIC cxx_std_20 PRIVATE c_std_17)
set_target_properties(ctorch PROPERTIES
    C_EXTENSIONS OFF CXX_EXTENSIONS OFF
    C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

generate_export_header(ctorch
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/dl/export.h"
    EXPORT_MACRO_NAME DL_API
    STATIC_DEFINE CTORCH_STATIC_DEFINE
)
configure_file(include/dl/version.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/dl/version.h" @ONLY)
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(ctorch PUBLIC CTORCH_STATIC_DEFINE)
endif()
target_include_directories(ctorch
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE src
)
if(MSVC)
    target_compile_options(ctorch PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>"
        PUBLIC "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>")
else()
    target_compile_options(ctorch PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic>")
    target_link_libraries(ctorch PRIVATE m)
endif()

if(CTORCH_USE_BLAS)
    set(BLA_SIZEOF_INTEGER 4)
    find_package(BLAS REQUIRED)
    find_path(CTORCH_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
    target_sources(ctorch PRIVATE src/backends/blas.cpp)
    target_include_directories(ctorch PRIVATE "${CTORCH_CBLAS_INCLUDE_DIR}")
    target_link_libraries(ctorch PRIVATE BLAS::BLAS)
    target_compile_definitions(ctorch PRIVATE CTORCH_USE_BLAS)
endif()

if(CTORCH_USE_CUDA)
    enable_language(CUDA)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    find_package(CUDAToolkit 12 REQUIRED)
    set_target_properties(ctorch PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    target_sources(ctorch PRIVATE
        src/backends/cuda.cpp
        src/backends/cuda_context.cpp
        src/kernels/cuda/pointwise.cu
        src/kernels/cuda/random.cu
        src/kernels/cuda/comparison.cu
        src/kernels/cuda/reduction.cu
        src/kernels/cuda/normalization.cu
        src/kernels/cuda/batch_normalization.cu
        src/kernels/cuda/matmul.cpp
        src/kernels/cuda/linear.cpp
        src/kernels/cuda/indexing.cu
        src/kernels/cuda/loss.cu
        src/kernels/cuda/optimizer.cu
        src/kernels/cuda/gradients.cu
        src/kernels/cuda/spatial.cu
        src/kernels/cuda/resampling.cu
    )
    target_compile_definitions(ctorch PRIVATE CTORCH_USE_CUDA)
    # 设备上下文与 cuBLASLt、cuDNN 计算方案使用标准互斥量管理共享状态。
    target_link_libraries(ctorch PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt Threads::Threads)
    if(MSVC)
        target_compile_options(ctorch PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
    endif()
endif()

if(CTORCH_USE_CUDNN)
    if(NOT CTORCH_USE_CUDA)
        message(FATAL_ERROR "CTORCH_USE_CUDNN requires CTORCH_USE_CUDA=ON")
    endif()
    find_package(CUDNN 9.7 REQUIRED)
    find_package(CUDNNFrontend 1.12 REQUIRED)
    target_sources(ctorch PRIVATE src/backends/cudnn.cpp)
    target_include_directories(ctorch SYSTEM PRIVATE "${CUDNNFrontend_INCLUDE_DIR}")
    target_link_libraries(ctorch PRIVATE CUDNN::CUDNN)
    target_compile_definitions(ctorch PRIVATE CTORCH_USE_CUDNN CUDNN_FRONTEND_SKIP_JSON_LIB)
endif()

if(CTORCH_USE_DLPACK)
    find_package(DLPack 1.0...<2 REQUIRED)
    target_sources(ctorch PRIVATE src/core/dlpack.cpp src/api/dlpack.cpp)
    target_link_libraries(ctorch PUBLIC DLPack::DLPack)
endif()

if(CTORCH_USE_NPU)
    add_subdirectory(extensions/npu)
endif()

configure_package_config_file(cmake/CtorchConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CtorchConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch"
)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/CtorchConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion
)
install(TARGETS ctorch EXPORT CtorchTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/dl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h"
    PATTERN "dlpack.h" EXCLUDE PATTERN "dlpack_c_api.h" EXCLUDE)
if(CTORCH_USE_DLPACK)
    install(FILES include/dl/dlpack.h include/dl/dlpack_c_api.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dl")
    install(FILES cmake/FindDLPack.cmake DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch")
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/dl/export.h"
              "${CMAKE_CURRENT_BINARY_DIR}/include/dl/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dl"
)
install(EXPORT CtorchTargets NAMESPACE ctorch:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch")
install(FILES COPYRIGHT LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Ctorch")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CtorchConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CtorchConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch"
)
if(CTORCH_USE_CUDNN)
    install(FILES cmake/FindCUDNN.cmake DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch")
endif()

if(CTORCH_BUILD_TESTS)
    enable_testing()
    if(CTORCH_DEVELOPMENT_DIR)
        add_subdirectory("${CTORCH_DEVELOPMENT_DIR}/tests" tests)
    else()
        add_subdirectory(tests)
    endif()
endif()

if(CTORCH_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CTORCH_BUILD_BENCHMARKS)
    add_subdirectory("${CTORCH_DEVELOPMENT_DIR}/benchmarks" benchmarks)
endif()
