#include "api/common.h"

#include <cstring>

using namespace dl::capi;

extern "C" {

void dl_tensor_release(dl_tensor* tensor) { delete tensor; }
dl_status dl_tensor_retain(const dl_tensor* tensor, dl_tensor** output) {
    return make_handle(output, [&] { return tensor->value; });
}
dl_tensor_info dl_tensor_get_info(const dl_tensor* tensor) {
    const auto& value = tensor->value;
    return {static_cast<dl_dtype>(value.dtype()), convert(value.device()), value.numel(),
        value.is_contiguous(), value.requires_grad(), value.is_leaf()};
}
dl_shape dl_tensor_shape(const dl_tensor* tensor) {
    const auto& shape = tensor->value.shape();
    return {shape.data(), shape.size()};
}
dl_shape dl_tensor_strides(const dl_tensor* tensor) {
    const auto& strides = tensor->value.strides();
    return {strides.data(), strides.size()};
}
const void* dl_tensor_data(const dl_tensor* tensor) { return tensor->value.data<std::byte>(); }
size_t dl_tensor_nbytes(const dl_tensor* tensor) {
    return static_cast<size_t>(tensor->value.numel()) * dl::element_size(tensor->value.dtype());
}
dl_status dl_tensor_copy_to_host(const dl_tensor* tensor, void* destination) {
    return invoke([&] {
        const auto bytes = dl_tensor_nbytes(tensor);
        if (bytes != 0) {
            auto dense = tensor->value.detach().to(dl::Device::cpu()).contiguous();
            std::memcpy(destination, dense.data<std::byte>(), bytes);
        }
    });
}
dl_status dl_from_host(const void* data, dl_shape shape, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::detail::from_host(data, convert(shape), convert(options.dtype),
            options.requires_grad != 0, convert(options.device));
    });
}

dl_status dl_backward(const dl_tensor* output, const dl_tensor* seed) {
    return invoke([&] {
        if (seed) {
            output->value.backward(seed->value);
        } else {
            output->value.backward();
        }
    });
}
dl_status dl_tensor_gradient(const dl_tensor* tensor, dl_tensor** output) {
    *output = nullptr;
    return invoke([&] {
        if (auto gradient = tensor->value.grad()) {
            *output = new dl_tensor{std::move(*gradient)};
        }
    });
}
dl_status dl_tensor_zero_grad(const dl_tensor* tensor) { return invoke([&] { tensor->value.zero_grad(); }); }
dl_status dl_tensor_set_requires_grad(const dl_tensor* tensor, int32_t enabled) {
    return invoke([&] { tensor->value.requires_grad_(enabled != 0); });
}
dl_status dl_no_grad_begin(dl_grad_scope** output) {
    return dl_grad_mode_begin(0, output);
}
void dl_no_grad_end(dl_grad_scope* scope) { dl_grad_mode_end(scope); }

dl_status dl_grad_mode_begin(int32_t enabled, dl_grad_scope** output) {
    *output = nullptr;
    return invoke([&] { *output = new dl_grad_scope(enabled != 0); });
}
void dl_grad_mode_end(dl_grad_scope* scope) { delete scope; }

dl_status dl_autocast_begin(dl_device_type device, dl_dtype dtype, int32_t enabled, dl_autocast_scope** output) {
    return make_handle(output, [&] { return dl::AutocastGuard(static_cast<dl::DeviceType>(device), convert(dtype), enabled != 0); });
}
void dl_autocast_end(dl_autocast_scope* scope) { delete scope; }
int32_t dl_grad_enabled(void) { return dl::is_grad_enabled(); }
dl_status dl_synchronize(dl_device device) { return invoke([&] { dl::synchronize(convert(device)); }); }

dl_status dl_generator_create(uint64_t seed, dl_generator** output) {
    return make_handle(output, [&] { return std::make_shared<dl::Generator>(seed); });
}
void dl_generator_release(dl_generator* generator) { delete generator; }
dl_status dl_generator_seed(dl_generator* generator, uint64_t seed) {
    return invoke([&] { generator->value->seed(seed); });
}
dl_status dl_manual_seed(uint64_t seed) { return invoke([&] { dl::manual_seed(seed); }); }

dl_random_key dl_random_fold_in(dl_random_key key, uint64_t label) {
    return convert(dl::fold_in(convert(key), label));
}

void dl_random_split(dl_random_key key, size_t count, dl_random_key* outputs) {
    for (size_t i = 0; i < count; ++i) outputs[i] = convert(dl::fold_in(convert(key), i));
}

dl_status dl_random_key_tensor(dl_random_key key, dl_tensor** output) {
    return make_handle(output, [&] { return dl::key_tensor(convert(key)); });
}

dl_status dl_random_key_from_tensor(const dl_tensor* tensor, dl_random_key* output) {
    return invoke([&] { *output = convert(dl::key_from_tensor(tensor->value)); });
}

dl_status dl_full(dl_shape shape, double value, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::full(convert(shape), value, convert(options)); });
}

dl_status dl_zeros(dl_shape shape, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::zeros(convert(shape), convert(options)); });
}

dl_status dl_ones(dl_shape shape, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::ones(convert(shape), convert(options)); });
}

dl_status dl_full_like(const dl_tensor* input, double value, int32_t requires_grad, dl_tensor** output) {
    return make_handle(output, [&] { return dl::full_like(input->value, value, requires_grad != 0); });
}
dl_status dl_zeros_like(const dl_tensor* input, int32_t requires_grad, dl_tensor** output) {
    return make_handle(output, [&] { return dl::zeros_like(input->value, requires_grad != 0); });
}
dl_status dl_ones_like(const dl_tensor* input, int32_t requires_grad, dl_tensor** output) {
    return make_handle(output, [&] { return dl::ones_like(input->value, requires_grad != 0); });
}
dl_status dl_arange(double start, double end, double step, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::arange(start, end, step, convert(options)); });
}
dl_status dl_linspace(double start, double end, int64_t steps, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::linspace(start, end, steps, convert(options)); });
}
dl_status dl_eye(int64_t rows, int64_t columns, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::eye(rows, columns, convert(options)); });
}

dl_status dl_clone(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.clone(); });
}

dl_status dl_detach(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.detach(); });
}

dl_status dl_contiguous(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.contiguous(); });
}

dl_status dl_astype(const dl_tensor* input, dl_dtype dtype, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.astype(convert(dtype)); });
}

dl_status dl_to(const dl_tensor* input, dl_device device, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.to(convert(device)); });
}

dl_status dl_view(const dl_tensor* input, dl_shape shape, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.view(convert(shape)); });
}

dl_status dl_reshape(const dl_tensor* input, dl_shape shape, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.reshape(convert(shape)); });
}

dl_status dl_flatten(const dl_tensor* input, int64_t first, int64_t last, dl_tensor** output) {
    return make_handle(output, [&] { return dl::flatten(input->value, first, last); });
}

dl_status dl_expand(const dl_tensor* input, dl_shape shape, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.expand(convert(shape)); });
}

dl_status dl_unsqueeze(const dl_tensor* input, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.unsqueeze(axis); });
}
dl_status dl_squeeze(const dl_tensor* input, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.squeeze(axis); });
}
dl_status dl_squeeze_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.squeeze(); });
}
dl_status dl_select(const dl_tensor* input, int64_t axis, int64_t index, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.select(axis, index); });
}
dl_status dl_narrow(const dl_tensor* input, int64_t axis, int64_t start, int64_t length, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.narrow(axis, start, length); });
}
dl_status dl_repeat(const dl_tensor* input, dl_shape repeats, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.repeat(convert(repeats)); });
}

dl_status dl_tensor_list_create(const dl_tensor* const* tensors, size_t count, dl_tensor_list** output) {
    return make_handle(output, [&] { return tensor_list(tensors, count); });
}
void dl_tensor_list_release(dl_tensor_list* list) { delete list; }
size_t dl_tensor_list_size(const dl_tensor_list* list) { return list->value.size(); }
dl_status dl_tensor_list_get(const dl_tensor_list* list, size_t index, dl_tensor** output) {
    return make_handle(output, [&] { return list->value[index]; });
}
dl_status dl_split(const dl_tensor* input, int64_t split_size, int64_t axis, dl_tensor_list** output) {
    return make_handle(output, [&] { return input->value.split(split_size, axis); });
}
dl_status dl_split_with_sizes(const dl_tensor* input, dl_shape sizes, int64_t axis, dl_tensor_list** output) {
    return make_handle(output, [&] { return input->value.split(convert(sizes), axis); });
}
dl_status dl_chunk(const dl_tensor* input, int64_t chunks, int64_t axis, dl_tensor_list** output) {
    return make_handle(output, [&] { return input->value.chunk(chunks, axis); });
}
dl_status dl_unbind(const dl_tensor* input, int64_t axis, dl_tensor_list** output) {
    return make_handle(output, [&] { return input->value.unbind(axis); });
}

dl_status dl_transpose(const dl_tensor* input, int64_t first, int64_t second, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.transpose(first, second); });
}

dl_status dl_permute(const dl_tensor* input, dl_shape axes, dl_tensor** output) {
    return make_handle(output, [&] { return input->value.permute(convert(axes)); });
}

dl_status dl_slice(const dl_tensor* input, int64_t axis, int64_t start, int64_t stop, int64_t step,
    dl_tensor** output) {
    return make_handle(output, [&] { return input->value.slice(axis, start, stop, step); });
}

dl_status dl_add(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::add);
}

dl_status dl_add_relu(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::add_relu);
}

dl_status dl_sub(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::sub);
}

dl_status dl_mul(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::mul);
}

dl_status dl_div(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::div);
}

dl_status dl_eq(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::eq);
}
dl_status dl_ne(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::ne);
}
dl_status dl_lt(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::lt);
}
dl_status dl_le(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::le);
}
dl_status dl_gt(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::gt);
}
dl_status dl_ge(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::ge);
}

dl_status dl_matmul(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return binary(a, b, output, dl::matmul);
}

dl_status dl_linear(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_tensor** output) {
    return make_handle(output, [&] { return dl::linear(input->value, weight->value, optional(bias)); });
}

dl_status dl_cross_entropy(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return make_handle(output, [&] { return dl::cross_entropy(a->value, b->value); });
}

dl_status dl_mse_loss(const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return make_handle(output, [&] { return dl::mse_loss(a->value, b->value); });
}

dl_status dl_cross_entropy_with_reduction(const dl_tensor* logits, const dl_tensor* labels,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::cross_entropy(logits->value, labels->value, convert(reduction)); });
}

dl_status dl_cross_entropy_ignore_index(const dl_tensor* logits, const dl_tensor* labels,
    dl_loss_reduction reduction, int64_t ignore_index, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::cross_entropy(logits->value, labels->value, convert(reduction), ignore_index);
    });
}

dl_status dl_mse_loss_with_reduction(const dl_tensor* prediction, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::mse_loss(prediction->value, target->value, convert(reduction)); });
}

dl_status dl_nll_loss(const dl_tensor* probabilities, const dl_tensor* labels,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::nll_loss(probabilities->value, labels->value, convert(reduction)); });
}

dl_status dl_nll_loss_ignore_index(const dl_tensor* probabilities, const dl_tensor* labels,
    dl_loss_reduction reduction, int64_t ignore_index, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::nll_loss(probabilities->value, labels->value, convert(reduction), ignore_index);
    });
}

dl_status dl_l1_loss(const dl_tensor* prediction, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::l1_loss(prediction->value, target->value, convert(reduction)); });
}

dl_status dl_smooth_l1_loss(const dl_tensor* prediction, const dl_tensor* target,
    double beta, dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::smooth_l1_loss(prediction->value, target->value, beta, convert(reduction)); });
}

dl_status dl_huber_loss(const dl_tensor* prediction, const dl_tensor* target,
    double delta, dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::huber_loss(prediction->value, target->value, delta, convert(reduction)); });
}

dl_status dl_binary_cross_entropy_with_logits(const dl_tensor* logits, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::binary_cross_entropy_with_logits(logits->value, target->value, convert(reduction));
    });
}

dl_status dl_binary_cross_entropy(const dl_tensor* probability, const dl_tensor* target,
    dl_loss_reduction reduction, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::binary_cross_entropy(probability->value, target->value, convert(reduction));
    });
}

dl_status dl_pow(const dl_tensor* base, const dl_tensor* exponent, dl_tensor** output) {
    return make_handle(output, [&] { return dl::pow(base->value, exponent->value); });
}
dl_status dl_pow_scalar(const dl_tensor* base, double exponent, dl_tensor** output) {
    return make_handle(output, [&] { return dl::pow(base->value, exponent); });
}
dl_status dl_scalar_pow(double base, const dl_tensor* exponent, dl_tensor** output) {
    return make_handle(output, [&] { return dl::pow(base, exponent->value); });
}
dl_status dl_clamp(const dl_tensor* input, double minimum, double maximum, dl_tensor** output) {
    return make_handle(output, [&] { return dl::clamp(input->value, minimum, maximum); });
}
dl_status dl_clamp_min(const dl_tensor* input, double minimum, dl_tensor** output) {
    return make_handle(output, [&] { return dl::clamp_min(input->value, minimum); });
}
dl_status dl_clamp_max(const dl_tensor* input, double maximum, dl_tensor** output) {
    return make_handle(output, [&] { return dl::clamp_max(input->value, maximum); });
}

dl_status dl_add_scalar(const dl_tensor* input, double value, dl_tensor** output) {
    return make_handle(output, [&] { return input->value + value; });
}

dl_status dl_mul_scalar(const dl_tensor* input, double value, dl_tensor** output) {
    return make_handle(output, [&] { return input->value * value; });
}

dl_status dl_neg(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::neg);
}

dl_status dl_exp(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::exp);
}

dl_status dl_log(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::log);
}

dl_status dl_log1p(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::log1p); }
dl_status dl_expm1(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::expm1); }
dl_status dl_abs(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::abs); }
dl_status dl_sin(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::sin); }
dl_status dl_cos(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::cos); }
dl_status dl_tan(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::tan); }
dl_status dl_asin(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::asin); }
dl_status dl_acos(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::acos); }
dl_status dl_atan(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::atan); }

dl_status dl_sqrt(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::sqrt);
}

dl_status dl_tanh(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::tanh);
}

dl_status dl_sigmoid(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::sigmoid);
}

dl_status dl_relu(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::relu);
}

dl_status dl_gelu(const dl_tensor* input, dl_tensor** output) {
    return unary(input, output, dl::gelu);
}

dl_status dl_softplus(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::softplus); }
dl_status dl_silu(const dl_tensor* input, dl_tensor** output) { return unary(input, output, dl::silu); }
dl_status dl_leaky_relu(const dl_tensor* input, double negative_slope, dl_tensor** output) {
    return make_handle(output, [&] { return dl::leaky_relu(input->value, negative_slope); });
}

dl_status dl_sum(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::sum(input->value, convert(axes), keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}

dl_status dl_sum_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::sum(input->value); });
}

dl_status dl_mean(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::mean(input->value, convert(axes), keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}

dl_status dl_mean_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::mean(input->value); });
}

dl_status dl_amax(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::amax(input->value, convert(axes), keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}

dl_status dl_amax_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::amax(input->value); });
}

dl_status dl_softmax(const dl_tensor* input, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] { return dl::softmax(input->value, axis); });
}

dl_status dl_log_softmax(const dl_tensor* input, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] { return dl::log_softmax(input->value, axis); });
}

dl_status dl_where(const dl_tensor* mask, const dl_tensor* a, const dl_tensor* b, dl_tensor** output) {
    return make_handle(output, [&] { return dl::where(mask->value, a->value, b->value); });
}

dl_status dl_index_select(const dl_tensor* input, int64_t axis, const dl_tensor* indices, dl_tensor** output) {
    return make_handle(output, [&] { return dl::index_select(input->value, axis, indices->value); });
}

dl_status dl_embedding(const dl_tensor* indices, const dl_tensor* weight, dl_tensor** output) {
    return binary(indices, weight, output, dl::embedding);
}

dl_status dl_gather(const dl_tensor* input, int64_t axis, const dl_tensor* indices, dl_tensor** output) {
    return make_handle(output, [&] { return dl::gather(input->value, axis, indices->value); });
}

dl_status dl_scatter_add(const dl_tensor* input, int64_t axis, const dl_tensor* indices,
    const dl_tensor* source, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::scatter_add(input->value, axis, indices->value, source->value);
    });
}

dl_status dl_concat(const dl_tensor* const* inputs, size_t count, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] {
        auto values = tensor_list(inputs, count);
        return dl::concat(values, axis);
    });
}

dl_status dl_stack(const dl_tensor* const* inputs, size_t count, int64_t axis, dl_tensor** output) {
    return make_handle(output, [&] {
        auto values = tensor_list(inputs, count);
        return dl::stack(values, axis);
    });
}

dl_status dl_conv2d(const dl_tensor* input, const dl_tensor* weight, const dl_tensor* bias,
    dl_conv2d_options options, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::conv2d(input->value, weight->value, optional(bias), convert(options));
    });
}

dl_status dl_max_pool2d(const dl_tensor* input, dl_pool2d_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::max_pool2d(input->value, convert(options)); });
}

dl_status dl_avg_pool2d(const dl_tensor* input, dl_pool2d_options options, int32_t count_include_pad,
    dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::avg_pool2d(input->value, convert(options), count_include_pad != 0);
    });
}

dl_status dl_layer_norm(const dl_tensor* input, dl_shape normalized_shape,
    const dl_tensor* weight, const dl_tensor* bias, double epsilon, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::layer_norm(input->value, convert(normalized_shape), optional(weight), optional(bias), epsilon);
    });
}

dl_status dl_dropout(const dl_tensor* input, double probability, int32_t training, dl_tensor** output) {
    return make_handle(output, [&] { return dl::dropout(input->value, probability, training != 0); });
}

dl_status dl_dropout_with_generator(const dl_tensor* input, double probability, int32_t training,
    dl_generator* generator, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::dropout(input->value, probability, random_generator(generator), training != 0);
    });
}

dl_status dl_dropout_with_key(const dl_tensor* input, double probability, int32_t training,
    dl_random_key key, dl_tensor** output) {
    return make_handle(output, [&] { return dl::dropout(input->value, probability, convert(key), training != 0); });
}

dl_status dl_scaled_dot_product_attention(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, double dropout, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::scaled_dot_product_attention(query->value, key->value, value->value,
            optional(keep_mask), dropout);
    });
}

dl_status dl_scaled_dot_product_attention_ex(const dl_tensor* query, const dl_tensor* key,
    const dl_tensor* value, const dl_tensor* keep_mask, dl_attention_compute_options options,
    const dl_random_key* random_key, dl_tensor** output) {
    return make_handle(output, [&] {
        return random_key
            ? dl::scaled_dot_product_attention(convert(options), query->value, key->value, value->value,
                convert(*random_key), optional(keep_mask))
            : dl::scaled_dot_product_attention(convert(options), query->value, key->value, value->value,
                optional(keep_mask));
    });
}

dl_status dl_causal_mask(int64_t query_length, int64_t key_length, dl_device device, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::causal_mask(query_length, key_length, convert(device));
    });
}

dl_status dl_uniform(dl_shape shape, double low, double high, dl_tensor_options options,
    dl_generator* generator, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::uniform(convert(shape), low, high, convert(options), random_generator(generator));
    });
}

dl_status dl_randn(dl_shape shape, dl_tensor_options options, dl_generator* generator, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::randn(convert(shape), convert(options), random_generator(generator));
    });
}

dl_status dl_bernoulli(dl_shape shape, double probability, dl_device device, dl_generator* generator,
    dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::bernoulli(convert(shape), probability, random_generator(generator), convert(device));
    });
}

dl_status dl_uniform_with_key(dl_random_key key, dl_shape shape, double low, double high,
    dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::uniform(convert(key), convert(shape), low, high, convert(options)); });
}

dl_status dl_randn_with_key(dl_random_key key, dl_shape shape, dl_tensor_options options, dl_tensor** output) {
    return make_handle(output, [&] { return dl::randn(convert(key), convert(shape), convert(options)); });
}

dl_status dl_bernoulli_with_key(dl_random_key key, dl_shape shape, double probability,
    dl_device device, dl_tensor** output) {
    return make_handle(output, [&] { return dl::bernoulli(convert(key), convert(shape), probability, convert(device)); });
}

} // extern "C"
