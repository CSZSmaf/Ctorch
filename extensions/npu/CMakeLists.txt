# NPU 扩展独立链接 OpenVINO，主库的头文件与安装依赖保持独立。
find_package(OpenVINO 2026.0 REQUIRED COMPONENTS Runtime)

add_library(ctorch_npu src/openvino.cpp src/c_api.cpp)
add_library(ctorch::npu ALIAS ctorch_npu)
target_compile_features(ctorch_npu PUBLIC cxx_std_20)
target_link_libraries(ctorch_npu PUBLIC ctorch::ctorch PRIVATE openvino::runtime)
target_include_directories(ctorch_npu PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
)
set_target_properties(ctorch_npu PROPERTIES
    EXPORT_NAME npu
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)
generate_export_header(ctorch_npu
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/dl/npu_export.h"
    EXPORT_MACRO_NAME DL_NPU_API
    STATIC_DEFINE CTORCH_NPU_STATIC_DEFINE
)
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(ctorch_npu PUBLIC CTORCH_NPU_STATIC_DEFINE)
endif()
if(MSVC)
    target_compile_options(ctorch_npu PRIVATE /W4)
else()
    target_compile_options(ctorch_npu PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS ctorch_npu EXPORT CtorchNpuTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/dl/npu.h include/dl/npu_c_api.h "${CMAKE_CURRENT_BINARY_DIR}/include/dl/npu_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dl"
)
install(EXPORT CtorchNpuTargets NAMESPACE ctorch::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch"
)

if(CTORCH_BUILD_EXAMPLES)
    add_executable(ctorch_npu_infer examples/infer.cpp)
    target_link_libraries(ctorch_npu_infer PRIVATE ctorch::npu)
    set_target_properties(ctorch_npu_infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    add_executable(ctorch_npu_c_infer examples/infer.c)
    target_compile_features(ctorch_npu_c_infer PRIVATE c_std_17)
    target_link_libraries(ctorch_npu_c_infer PRIVATE ctorch::npu)
    set_target_properties(ctorch_npu_c_infer PROPERTIES
        C_EXTENSIONS OFF LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
    )
endif()
