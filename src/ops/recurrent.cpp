#include "ops/recurrent.h"

namespace dl::detail {

RecurrentState recurrent_step(RecurrentKind kind, const Tensor& projected, const Tensor& recurrent,
                               const Tensor& hidden, std::optional<Tensor> cell) {
    if (kind == RecurrentKind::gru) {
        auto input_gates = projected.chunk(3, -1);
        auto hidden_gates = recurrent.chunk(3, -1);
        auto reset = sigmoid(input_gates[0] + hidden_gates[0]);
        auto update = sigmoid(input_gates[1] + hidden_gates[1]);
        auto candidate = tanh(input_gates[2] + reset * hidden_gates[2]);
        return {(1.0 - update) * candidate + update * hidden, std::nullopt};
    }
    auto combined = projected + recurrent;
    if (kind == RecurrentKind::lstm) {
        auto gates = combined.chunk(4, -1);
        auto input_gate = sigmoid(gates[0]);
        auto forget_gate = sigmoid(gates[1]);
        auto candidate = tanh(gates[2]);
        auto output_gate = sigmoid(gates[3]);
        auto next_cell = forget_gate * *cell + input_gate * candidate;
        return {output_gate * tanh(next_cell), std::move(next_cell)};
    }
    return {kind == RecurrentKind::rnn_relu ? relu(combined) : tanh(combined), std::nullopt};
}

} // namespace dl::detail

namespace dl {

Tensor rnn_cell(const Tensor& input, const Tensor& hidden, const Tensor& weight_ih, const Tensor& weight_hh,
                std::optional<Tensor> bias_ih, std::optional<Tensor> bias_hh, RNNActivation activation) {
    auto combined = linear(input, weight_ih, std::move(bias_ih)) + linear(hidden, weight_hh, std::move(bias_hh));
    return activation == RNNActivation::relu ? relu(combined) : tanh(combined);
}

Tensor gru_cell(const Tensor& input, const Tensor& hidden, const Tensor& weight_ih, const Tensor& weight_hh,
                std::optional<Tensor> bias_ih, std::optional<Tensor> bias_hh) {
    return detail::recurrent_step(RecurrentKind::gru, linear(input, weight_ih, std::move(bias_ih)),
        linear(hidden, weight_hh, std::move(bias_hh)), hidden, std::nullopt).hidden;
}

LSTMState lstm_cell(const Tensor& input, const LSTMState& state, const Tensor& weight_ih, const Tensor& weight_hh,
                    std::optional<Tensor> bias_ih, std::optional<Tensor> bias_hh) {
    auto next = detail::recurrent_step(RecurrentKind::lstm, linear(input, weight_ih, std::move(bias_ih)),
        linear(state.hidden, weight_hh, std::move(bias_hh)), state.hidden, state.cell);
    return {std::move(next.hidden), std::move(*next.cell)};
}

} // namespace dl
