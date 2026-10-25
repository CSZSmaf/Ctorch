#include "backends/backend.h"

#include "backends/cuda_context.h"
#include "kernels/cuda/ops.h"

#ifdef CTORCH_USE_CUDNN
#include "backends/cudnn.h"
#endif

namespace dl::detail {

const Backend& cuda_backend() {
    static const Backend implementation{
        .allocate = cuda::allocate,
        .synchronize = cuda::synchronize,
        .transfer = cuda::transfer,
        .random = cuda::random,
        .dropout = cuda::dropout,
        .random_keyed = cuda::random_keyed,
        .dropout_keyed = cuda::dropout_keyed,
        .masked_scale = cuda::masked_scale,
        .full = cuda::full,
        .clone = cuda::clone,
        .cast = cuda::cast,
        .copy_into = cuda::copy_into,
        .axpy = cuda::axpy,
        .unary = cuda::unary,
        .unary_backward_into = cuda::unary_backward_into,
        .add = cuda::add,
        .add_relu = cuda::add_relu,
        .sub = cuda::sub,
        .mul = cuda::mul,
        .div = cuda::div,
        .pow = cuda::pow,
        .scale_into = cuda::scale_into,
        .where = cuda::where,
        .compare = cuda::compare,
        .sum = cuda::sum,
        .sum_into = cuda::sum_into,
        .amax = cuda::amax,
        .arg_extreme = cuda::arg_extreme,
        .matmul_into = cuda::matmul_into,
        .linear_into = cuda::linear_into,
        .softmax = cuda::softmax,
        .softmax_backward = cuda::softmax_backward,
        .normalization = cuda::normalization,
        .normalization_backward = cuda::normalization_backward,
        .gather = cuda::gather,
        .repeat = cuda::repeat,
        .repeat_backward_into = cuda::repeat_backward_into,
        .scatter_add_into = cuda::scatter_add_into,
        .cross_entropy = cuda::cross_entropy,
        .cross_entropy_backward_into = cuda::cross_entropy_backward_into,
        .momentum = cuda::momentum,
        .unscale_gradients = cuda::unscale_gradients,
        .clip_grad_norm = cuda::clip_grad_norm,
        .clip_grad_value = cuda::clip_grad_value,
        .sgd = cuda::sgd,
        .adam = cuda::adam,
        .rmsprop = cuda::rmsprop,
        .adagrad = cuda::adagrad,
        .im2col = cuda::im2col,
        .col2im_into = cuda::col2im_into,
        .max_pool2d = cuda::max_pool2d,
        .max_pool2d_backward_into = cuda::max_pool2d_backward_into,
        .avg_pool2d = cuda::avg_pool2d,
        .avg_pool2d_backward_into = cuda::avg_pool2d_backward_into,
        .adaptive_avg_pool2d = cuda::adaptive_avg_pool2d,
        .adaptive_avg_pool2d_backward_into = cuda::adaptive_avg_pool2d_backward_into,
        .interpolate = cuda::interpolate,
        .interpolate_backward_into = cuda::interpolate_backward_into,
#ifdef CTORCH_USE_CUDNN
        .convolution = {.prepare = cuda::prepare_convolution, .execute = cuda::convolution}
#else
        .convolution = {}
#endif
    };
    return implementation;
}

} // namespace dl::detail
