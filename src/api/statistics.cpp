#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_status dl_amin(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::amin(input->value, convert(axes), keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_amin_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::amin(input->value); });
}

dl_status dl_argmax(const dl_tensor* input, int64_t axis, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::argmax(input->value, axis, keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_argmax_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::argmax(input->value); });
}
dl_status dl_argmin(const dl_tensor* input, int64_t axis, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::argmin(input->value, axis, keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_argmin_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::argmin(input->value); });
}

dl_status dl_logsumexp(const dl_tensor* input, dl_shape axes, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::logsumexp(input->value, convert(axes), keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_logsumexp_all(const dl_tensor* input, dl_tensor** output) {
    return make_handle(output, [&] { return dl::logsumexp(input->value); });
}

dl_status dl_var(const dl_tensor* input, dl_shape axes, double correction, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::var(input->value, convert(axes), correction, keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_var_all(const dl_tensor* input, double correction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::var(input->value, correction); });
}
dl_status dl_stddev(const dl_tensor* input, dl_shape axes, double correction, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::stddev(input->value, convert(axes), correction, keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_stddev_all(const dl_tensor* input, double correction, dl_tensor** output) {
    return make_handle(output, [&] { return dl::stddev(input->value, correction); });
}

dl_status dl_vector_norm(const dl_tensor* input, dl_shape axes, double order, int32_t keepdim, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::vector_norm(input->value, convert(axes), order, keepdim ? dl::KeepDim::yes : dl::KeepDim::no);
    });
}
dl_status dl_vector_norm_all(const dl_tensor* input, double order, dl_tensor** output) {
    return make_handle(output, [&] { return dl::vector_norm(input->value, order); });
}

} // extern "C"
