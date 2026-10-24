# cuDNN 的头文件和链接库来自调用方选定的安装目录。
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT
    PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(CUDNN_LIBRARY NAMES cudnn cudnn64_9 HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu)

if(CUDNN_INCLUDE_DIR)
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" _cudnn_version_lines
        REGEX "^#define CUDNN_(MAJOR|MINOR|PATCHLEVEL)[ \t]+[0-9]+")
    foreach(_component MAJOR MINOR PATCHLEVEL)
        string(REGEX MATCH "CUDNN_${_component}[ \t]+([0-9]+)" _match "${_cudnn_version_lines}")
        set(_cudnn_${_component} "${CMAKE_MATCH_1}")
    endforeach()
    set(CUDNN_VERSION "${_cudnn_MAJOR}.${_cudnn_MINOR}.${_cudnn_PATCHLEVEL}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN REQUIRED_VARS CUDNN_INCLUDE_DIR CUDNN_LIBRARY
    VERSION_VAR CUDNN_VERSION)
if(CUDNN_FOUND AND NOT TARGET CUDNN::CUDNN)
    add_library(CUDNN::CUDNN UNKNOWN IMPORTED)
    set_target_properties(CUDNN::CUDNN PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()
mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
