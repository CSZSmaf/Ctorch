set(prefix "${root}/test-install")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${root}" --config "${config}" --prefix "${prefix}"
    COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET)
set(options -G "${generator}" "-DCMAKE_BUILD_TYPE=${config}"
    "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DCMAKE_C_COMPILER=${c_compiler}" "-DCMAKE_CXX_COMPILER=${cxx_compiler}")
if(platform)
    list(APPEND options -A "${platform}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${root}/test-consumer" ${options}
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${root}/test-consumer" --config "${config}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${root}/test-consumer" -C "${config}" --output-on-failure
    COMMAND_ERROR_IS_FATAL ANY)
