# DLPack 只有 ABI 头文件，使用调用方选定的源码或安装目录。
find_path(DLPack_INCLUDE_DIR dlpack/dlpack.h HINTS "${DLPack_ROOT}" ENV DLPack_ROOT PATH_SUFFIXES include)
if(DLPack_INCLUDE_DIR)
    file(STRINGS "${DLPack_INCLUDE_DIR}/dlpack/dlpack.h" _dlpack_version_lines
        REGEX "^#define DLPACK_(MAJOR|MINOR)_VERSION[ \t]+[0-9]+")
    foreach(_component MAJOR MINOR)
        string(REGEX MATCH "DLPACK_${_component}_VERSION[ \t]+([0-9]+)" _match "${_dlpack_version_lines}")
        set(_dlpack_${_component} "${CMAKE_MATCH_1}")
    endforeach()
    set(DLPack_VERSION "${_dlpack_MAJOR}.${_dlpack_MINOR}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DLPack REQUIRED_VARS DLPack_INCLUDE_DIR
    VERSION_VAR DLPack_VERSION HANDLE_VERSION_RANGE)
if(DLPack_FOUND AND NOT TARGET DLPack::DLPack)
    add_library(DLPack::DLPack INTERFACE IMPORTED)
    set_target_properties(DLPack::DLPack PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${DLPack_INCLUDE_DIR}")
endif()
mark_as_advanced(DLPack_INCLUDE_DIR)
