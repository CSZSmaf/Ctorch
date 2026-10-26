add_library(ctorch_onnx src/export.cpp src/c_api.cpp)
add_library(ctorch::onnx ALIAS ctorch_onnx)
target_compile_features(ctorch_onnx PUBLIC cxx_std_20)
target_link_libraries(ctorch_onnx PUBLIC ctorch::ctorch)
target_include_directories(ctorch_onnx PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
)
set_target_properties(ctorch_onnx PROPERTIES EXPORT_NAME onnx CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
generate_export_header(ctorch_onnx
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/dl/onnx_export.h"
    EXPORT_MACRO_NAME DL_ONNX_API STATIC_DEFINE CTORCH_ONNX_STATIC_DEFINE)
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(ctorch_onnx PUBLIC CTORCH_ONNX_STATIC_DEFINE)
endif()
if(MSVC)
    target_compile_options(ctorch_onnx PRIVATE /W4)
else()
    target_compile_options(ctorch_onnx PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS ctorch_onnx EXPORT CtorchOnnxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/dl/onnx.h include/dl/onnx_c_api.h "${CMAKE_CURRENT_BINARY_DIR}/include/dl/onnx_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dl")
install(EXPORT CtorchOnnxTargets NAMESPACE ctorch:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Ctorch")
