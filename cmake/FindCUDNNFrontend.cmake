# frontend 为仅头文件的构建依赖，安装后的调用方只需要 cuDNN 运行库。
find_path(CUDNNFrontend_INCLUDE_DIR cudnn_frontend.h
    HINTS "${CUDNNFrontend_ROOT}" ENV CUDNNFrontend_ROOT PATH_SUFFIXES include)
if(CUDNNFrontend_INCLUDE_DIR)
    file(STRINGS "${CUDNNFrontend_INCLUDE_DIR}/cudnn_frontend_version.h" _frontend_version_lines
        REGEX "^#define CUDNN_FRONTEND_(MAJOR|MINOR|PATCH)_VERSION[ \t]+[0-9]+")
    foreach(_component MAJOR MINOR PATCH)
        string(REGEX MATCH "CUDNN_FRONTEND_${_component}_VERSION[ \t]+([0-9]+)" _match "${_frontend_version_lines}")
        set(_frontend_${_component} "${CMAKE_MATCH_1}")
    endforeach()
    set(CUDNNFrontend_VERSION "${_frontend_MAJOR}.${_frontend_MINOR}.${_frontend_PATCH}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNNFrontend REQUIRED_VARS CUDNNFrontend_INCLUDE_DIR
    VERSION_VAR CUDNNFrontend_VERSION)
mark_as_advanced(CUDNNFrontend_INCLUDE_DIR)
