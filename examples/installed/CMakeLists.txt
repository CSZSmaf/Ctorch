cmake_minimum_required(VERSION 3.24)
project(CtorchInstalledExamples LANGUAGES C CXX)

option(CTORCH_INSTALLED_NPU_EXAMPLES "Build the optional NPU C/C++ consumers" OFF)
if(CTORCH_INSTALLED_NPU_EXAMPLES)
    find_package(Ctorch 0.2 CONFIG REQUIRED COMPONENTS npu)
else()
    find_package(Ctorch 0.2 CONFIG REQUIRED)
endif()

include(../targets.cmake)

foreach(example IN LISTS CTORCH_C_EXAMPLES)
    ctorch_add_example(${example} ${example}.c)
endforeach()

foreach(example IN LISTS CTORCH_CPP_EXAMPLES)
    ctorch_add_example(${example} ${example}.cpp)
endforeach()

if(Ctorch_USE_CUDA)
    ctorch_add_example(cuda_memory cuda_memory.cpp)
endif()

if(Ctorch_USE_CUDNN)
    ctorch_add_example(cudnn_attention cudnn_attention.cpp)
    ctorch_add_example(c_cudnn_attention c_cudnn_attention.c)
endif()

if(Ctorch_USE_DLPACK)
    ctorch_add_example(dlpack_vjp dlpack_vjp.cpp)
    ctorch_add_example(c_dlpack c_dlpack.c)
endif()

if(CTORCH_INSTALLED_NPU_EXAMPLES)
    add_executable(npu_infer ../../extensions/npu/examples/infer.cpp)
    target_link_libraries(npu_infer PRIVATE ctorch::npu)
    add_executable(npu_c_infer ../../extensions/npu/examples/infer.c)
    target_compile_features(npu_c_infer PRIVATE c_std_17)
    set_target_properties(npu_c_infer PROPERTIES C_EXTENSIONS OFF LINKER_LANGUAGE CXX)
    target_link_libraries(npu_c_infer PRIVATE ctorch::npu)
endif()
