foreach(suite IN ITEMS tensor gradients state)
    add_executable(ctorch_check_${suite} ${suite}.cpp)
    target_link_libraries(ctorch_check_${suite} PRIVATE ctorch::ctorch)
    set_target_properties(ctorch_check_${suite} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    add_test(NAME ${suite} COMMAND ctorch_check_${suite})
endforeach()

add_executable(ctorch_check_c_api c_api.c)
target_link_libraries(ctorch_check_c_api PRIVATE ctorch::ctorch)
target_compile_features(ctorch_check_c_api PRIVATE c_std_17)
set_target_properties(ctorch_check_c_api PROPERTIES C_EXTENSIONS OFF LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
add_test(NAME c_api COMMAND ctorch_check_c_api)

add_test(NAME installed_package COMMAND ${CMAKE_COMMAND}
    "-Dsource=${CMAKE_CURRENT_SOURCE_DIR}/installed"
    "-Droot=${PROJECT_BINARY_DIR}"
    "-Dconfig=$<CONFIG>"
    "-Dgenerator=${CMAKE_GENERATOR}"
    "-Dplatform=${CMAKE_GENERATOR_PLATFORM}"
    "-Dc_compiler=${CMAKE_C_COMPILER}"
    "-Dcxx_compiler=${CMAKE_CXX_COMPILER}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/installed/run.cmake")
