#include "api/common.h"

using namespace dl::capi;

extern "C" {

dl_recurrent_options dl_default_recurrent_options(void) {
    const dl::RecurrentOptions value;
    return {value.layers, value.bias, value.batch_first, value.bidirectional, value.dropout};
}

dl_status dl_rnn_cell(const dl_tensor* input, const dl_tensor* hidden,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_rnn_activation activation, dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::rnn_cell(input->value, hidden->value, weight_ih->value, weight_hh->value,
            optional(bias_ih), optional(bias_hh), convert(activation));
    });
}
dl_status dl_gru_cell(const dl_tensor* input, const dl_tensor* hidden,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_tensor** output) {
    return make_handle(output, [&] {
        return dl::gru_cell(input->value, hidden->value, weight_ih->value, weight_hh->value,
            optional(bias_ih), optional(bias_hh));
    });
}
dl_status dl_lstm_cell(const dl_tensor* input, const dl_tensor* hidden, const dl_tensor* cell,
    const dl_tensor* weight_ih, const dl_tensor* weight_hh, const dl_tensor* bias_ih, const dl_tensor* bias_hh,
    dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto result = dl::lstm_cell(input->value, {hidden->value, cell->value}, weight_ih->value, weight_hh->value,
            optional(bias_ih), optional(bias_hh));
        return std::vector<dl::Tensor>{std::move(result.hidden), std::move(result.cell)};
    });
}

dl_status dl_rnn_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_rnn_activation activation, dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::RNN>(input_size, hidden_size, convert(recurrent), convert(activation), convert(options));
    });
}
dl_status dl_gru_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::GRU>(input_size, hidden_size, convert(recurrent), convert(options));
    });
}
dl_status dl_lstm_create(int64_t input_size, int64_t hidden_size, dl_recurrent_options recurrent,
    dl_tensor_options options, dl_module** output) {
    return make_handle(output, [&] {
        return std::make_shared<dl::nn::LSTM>(input_size, hidden_size, convert(recurrent), convert(options));
    });
}
dl_status dl_recurrent_forward(dl_module* module, const dl_tensor* input,
    const dl_tensor* hidden, const dl_tensor* cell, dl_tensor_list** output) {
    return make_handle(output, [&] {
        auto result = static_cast<dl::nn::Recurrent&>(*module->value).run(input->value, optional(hidden), optional(cell));
        std::vector<dl::Tensor> tensors{std::move(result.output), std::move(result.hidden)};
        if (result.cell) tensors.push_back(std::move(*result.cell));
        return tensors;
    });
}

} // extern "C"
