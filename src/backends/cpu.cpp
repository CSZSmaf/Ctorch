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
        allocate, synchronize, transfer,
        cpu::random, cpu::dropout, cpu::random_keyed, cpu::dropout_keyed, cpu::masked_scale,
        cpu::full, cpu::clone, cpu::cast, cpu::copy_into, cpu::axpy,
        cpu::unary, cpu::unary_backward, cpu::add, cpu::add_relu, cpu::sub, cpu::mul, cpu::div, cpu::pow,
        cpu::scale, cpu::where, cpu::compare, cpu::sum, cpu::sum_into, cpu::amax, cpu::arg_extreme, cpu::matmul_into, cpu::softmax,
        cpu::softmax_backward, cpu::normalization, cpu::normalization_backward,
        cpu::gather, cpu::repeat, cpu::repeat_backward_into, cpu::scatter_add_into,
        cpu::cross_entropy, cpu::cross_entropy_backward_into,
        cpu::momentum, cpu::unscale_gradients, cpu::clip_grad_norm, cpu::clip_grad_value,
        cpu::sgd, cpu::adam, cpu::rmsprop, cpu::adagrad,
        cpu::im2col, cpu::col2im_into, cpu::max_pool2d, cpu::max_pool2d_backward_into,
        cpu::avg_pool2d, cpu::avg_pool2d_backward_into,
        cpu::adaptive_avg_pool2d, cpu::adaptive_avg_pool2d_backward_into, cpu::interpolate, cpu::interpolate_backward_into,
        {}
    };
    return implementation;
}

} // namespace dl::detail
