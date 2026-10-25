#include "backends/backend.h"

#include "kernels/cpu/ops.h"

#include <cstring>
#include <new>

namespace dl::detail {
namespace {

Allocation allocate(std::size_t bytes, Device) {
    return {::operator new(bytes), [](void* address, void*) noexcept { ::operator delete(address); }, {}};
}
void synchronize(Device) {}
void transfer(void* target, Device, const void* source, Device, std::size_t bytes) {
    std::memmove(target, source, bytes);
}

} // namespace

const Backend& cpu_backend() {
    static const Backend implementation{
        .allocate = allocate,
        .synchronize = synchronize,
        .transfer = transfer,
        .random = cpu::random,
        .dropout = cpu::dropout,
        .random_keyed = cpu::random_keyed,
        .dropout_keyed = cpu::dropout_keyed,
        .masked_scale = cpu::masked_scale,
        .full = cpu::full,
        .clone = cpu::clone,
        .cast = cpu::cast,
        .copy_into = cpu::copy_into,
        .axpy = cpu::axpy,
        .unary = cpu::unary,
        .unary_backward_into = cpu::unary_backward_into,
        .add = cpu::add,
        .add_relu = cpu::add_relu,
        .sub = cpu::sub,
        .mul = cpu::mul,
        .div = cpu::div,
        .pow = cpu::pow,
        .scale_into = cpu::scale_into,
        .where = cpu::where,
        .compare = cpu::compare,
        .sum = cpu::sum,
        .sum_into = cpu::sum_into,
        .amax = cpu::amax,
        .arg_extreme = cpu::arg_extreme,
        .matmul_into = cpu::matmul_into,
        .softmax = cpu::softmax,
        .softmax_backward = cpu::softmax_backward,
        .normalization = cpu::normalization,
        .normalization_backward = cpu::normalization_backward,
        .gather = cpu::gather,
        .repeat = cpu::repeat,
        .repeat_backward_into = cpu::repeat_backward_into,
        .scatter_add_into = cpu::scatter_add_into,
        .cross_entropy = cpu::cross_entropy,
        .binary_logits = cpu::binary_logits,
        .binary_logits_backward_into = cpu::binary_logits_backward_into,
        .cross_entropy_backward_into = cpu::cross_entropy_backward_into,
        .momentum = cpu::momentum,
        .unscale_gradients = cpu::unscale_gradients,
        .clip_grad_norm = cpu::clip_grad_norm,
        .clip_grad_value = cpu::clip_grad_value,
        .sgd = cpu::sgd,
        .adam = cpu::adam,
        .rmsprop = cpu::rmsprop,
        .adagrad = cpu::adagrad,
        .im2col = cpu::im2col,
        .col2im_into = cpu::col2im_into,
        .max_pool2d = cpu::max_pool2d,
        .max_pool2d_backward_into = cpu::max_pool2d_backward_into,
        .avg_pool2d = cpu::avg_pool2d,
        .avg_pool2d_backward_into = cpu::avg_pool2d_backward_into,
        .adaptive_avg_pool2d = cpu::adaptive_avg_pool2d,
        .adaptive_avg_pool2d_backward_into = cpu::adaptive_avg_pool2d_backward_into,
        .interpolate = cpu::interpolate,
        .interpolate_backward_into = cpu::interpolate_backward_into,
        .convolution = {}
    };
    return implementation;
}

} // namespace dl::detail
