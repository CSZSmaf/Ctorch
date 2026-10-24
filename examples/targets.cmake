# 两种构建入口共用公开接口示例；真实数据训练入口由源码工程单独加入。
set(CTORCH_CPP_EXAMPLES
    first_training parameter_layer custom_layer custom_vjp activation_checkpoint gradients execution_plan multi_output_plan
    functional_state random_key functional_optimizer functional_module finetune)
set(CTORCH_C_EXAMPLES
    c_train c_custom_vjp c_activation_checkpoint c_execution_plan c_gradients
    c_gradient_processing c_batch_norm c_functional_optimizer c_functional_module)

# 目标名称由调用入口选择，源码位置始终相对本清单；库由源码或安装包提供。
function(ctorch_add_example target source)
    add_executable(${target} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/${source}")
    target_link_libraries(${target} PRIVATE ctorch::ctorch)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(source MATCHES "[.]c$")
        target_compile_features(${target} PRIVATE c_std_17)
        set_target_properties(${target} PROPERTIES C_EXTENSIONS OFF LINKER_LANGUAGE CXX)
    endif()
endfunction()
