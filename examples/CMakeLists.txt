include(targets.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

foreach(example IN ITEMS ${CTORCH_CPP_EXAMPLES} mnist_mlp mnist_cnn byte_transformer recurrent_train)
    ctorch_add_example(ctorch_${example} ${example}.cpp)
endforeach()

if(CTORCH_USE_CUDA)
    ctorch_add_example(ctorch_cuda_memory cuda_memory.cpp)
endif()

if(CTORCH_USE_CUDNN)
    ctorch_add_example(ctorch_cudnn_attention cudnn_attention.cpp)
    ctorch_add_example(ctorch_c_cudnn_attention c_cudnn_attention.c)
endif()

if(CTORCH_USE_DLPACK)
    ctorch_add_example(ctorch_dlpack_vjp dlpack_vjp.cpp)
    ctorch_add_example(ctorch_c_dlpack c_dlpack.c)
endif()

foreach(example IN LISTS CTORCH_C_EXAMPLES)
    ctorch_add_example(ctorch_${example} ${example}.c)
endforeach()
