#include "backends/backend.h"

#include "backends/cuda_context.h"
#include "kernels/cuda/ops.h"

#ifdef CTORCH_USE_CUDNN
#include "backends/cudnn.h"
#endif

namespace dl::detail {

const Backend& cuda_backend() {
    static const Backend implementation{
        cuda::allocate, cuda::synchronize, cuda::transfer,
        cuda::random, cuda::dropout, cuda::random_keyed, cuda::dropout_keyed, cuda::masked_scale,
        cuda::full, cuda::clone, cuda::cast, cuda::copy_into, cuda::axpy,
        cuda::unary, cuda::unary_backward, cuda::add, cuda::add_relu, cuda::sub, cuda::mul, cuda::div, cuda::pow,
        cuda::scale, cuda::where, cuda::compare, cuda::sum, cuda::sum_into, cuda::amax, cuda::arg_extreme, cuda::matmul_into, cuda::softmax,
        cuda::softmax_backward, cuda::normalization, cuda::normalization_backward,
        cuda::gather, cuda::repeat, cuda::repeat_backward_into, cuda::scatter_add_into,
        cuda::cross_entropy, cuda::cross_entropy_backward_into,
        cuda::momentum, cuda::unscale_gradients, cuda::clip_grad_norm, cuda::clip_grad_value,
        cuda::sgd, cuda::adam, cuda::rmsprop, cuda::adagrad,
        cuda::im2col, cuda::col2im_into, cuda::max_pool2d, cuda::max_pool2d_backward_into,
        cuda::avg_pool2d, cuda::avg_pool2d_backward_into,
        cuda::adaptive_avg_pool2d, cuda::adaptive_avg_pool2d_backward_into, cuda::interpolate, cuda::interpolate_backward_into,
#ifdef CTORCH_USE_CUDNN
        {cuda::prepare_convolution, cuda::convolution}
#else
        {}
#endif
    };
    return implementation;
}

} // namespace dl::detail
