#include "dl/nn.h"

#include "nn/initialization.h"
#include "ops/recurrent.h"

#include <algorithm>

namespace dl::nn {

Recurrent::Recurrent(Index input_size, Index hidden_size, RecurrentKind kind,
                      RecurrentOptions recurrent, TensorOptions options)
    : hidden_size_(hidden_size), kind_(kind), options_(recurrent) {
    const Index gates = kind == RecurrentKind::lstm ? 4 : kind == RecurrentKind::gru ? 3 : 1;
    const Index directions = recurrent.bidirectional ? 2 : 1;
    weights_.reserve(static_cast<std::size_t>(recurrent.layers * directions));
    for (Index layer = 0; layer < recurrent.layers; ++layer) {
        const Index features = layer == 0 ? input_size : hidden_size * directions;
        for (Index direction = 0; direction < directions; ++direction) {
            const auto suffix = "_l" + std::to_string(layer) + (direction == 0 ? "" : "_reverse");
            auto parameter = [&](const char* name, Shape shape) {
                return register_parameter(name + suffix, detail::uniform_parameter(std::move(shape), hidden_size, options));
            };
            Weights weights{parameter("weight_ih", {gates * hidden_size, features}),
                parameter("weight_hh", {gates * hidden_size, hidden_size}), std::nullopt, std::nullopt};
            if (recurrent.bias) {
                weights.input_bias = parameter("bias_ih", {gates * hidden_size});
                weights.hidden_bias = parameter("bias_hh", {gates * hidden_size});
            }
            weights_.push_back(std::move(weights));
        }
    }
}

Tensor Recurrent::forward(const Tensor& input) { return run(input).output; }

RecurrentResult Recurrent::run(const Tensor& input, std::optional<Tensor> hidden, std::optional<Tensor> cell) {
    const bool unbatched = input.ndim() == 2;
    auto sequence = unbatched ? input.unsqueeze(1) : options_.batch_first ? input.transpose(0, 1) : input;
    const Index length = sequence.size(0), batch = sequence.size(1);
    const Index directions = options_.bidirectional ? 2 : 1;
    const Shape state_shape{options_.layers * directions, batch, hidden_size_};
    auto initial = [&](const std::optional<Tensor>& state) {
        return state ? (unbatched ? state->unsqueeze(1) : *state)
                     : zeros(state_shape, {.dtype = input.dtype(), .device = input.device()});
    };
    auto initial_hidden = initial(hidden);
    std::optional<Tensor> initial_cell = kind_ == RecurrentKind::lstm ? std::optional{initial(cell)} : std::nullopt;
    std::vector<Tensor> final_hidden, final_cell;
    final_hidden.reserve(weights_.size());
    if (initial_cell) final_cell.reserve(weights_.size());

    for (Index layer = 0; layer < options_.layers; ++layer) {
        std::vector<Tensor> directional_outputs;
        directional_outputs.reserve(static_cast<std::size_t>(directions));
        for (Index direction = 0; direction < directions; ++direction) {
            const auto index = static_cast<std::size_t>(layer * directions + direction);
            const auto& source = weights_[index];
            // 每个方向只绑定一次，时间循环直接使用本次调用的参数句柄。
            const Weights weights{bound(source.input), bound(source.hidden),
                bound(source.input_bias), bound(source.hidden_bias)};
            auto h = initial_hidden.select(0, static_cast<Index>(index));
            std::optional<Tensor> c = initial_cell ? std::optional{initial_cell->select(0, static_cast<Index>(index))} : std::nullopt;

            // 整段输入共用一次 GEMM；每个时间步只计算依赖上一隐藏状态的投影。
            auto projected = linear(sequence.reshape({length * batch, sequence.size(2)}), weights.input, weights.input_bias)
                .reshape({length, batch, weights.input.size(0)});
            std::vector<Tensor> outputs;
            outputs.reserve(static_cast<std::size_t>(length));
            for (Index step = 0; step < length; ++step) {
                const Index time = direction == 0 ? step : length - 1 - step;
                auto next = detail::recurrent_step(kind_, projected.select(0, time),
                    linear(h, weights.hidden, weights.hidden_bias), h, c);
                h = std::move(next.hidden);
                c = std::move(next.cell);
                outputs.push_back(h);
            }
            final_hidden.push_back(h);
            if (c) final_cell.push_back(*c);
            // 反向遍历的结果重新按输入时间排列，末状态仍对应该方向最后处理的位置。
            if (direction != 0) std::reverse(outputs.begin(), outputs.end());
            directional_outputs.push_back(stack(outputs, 0));
        }
        sequence = directions == 1 ? std::move(directional_outputs[0]) : concat(directional_outputs, 2);
        if (layer + 1 < options_.layers) sequence = apply_dropout(sequence, options_.dropout);
    }

    auto h = stack(final_hidden, 0);
    std::optional<Tensor> c = initial_cell ? std::optional{stack(final_cell, 0)} : std::nullopt;
    if (unbatched) {
        sequence = sequence.squeeze(1);
        h = h.squeeze(1);
        if (c) c = c->squeeze(1);
    } else if (options_.batch_first) {
        sequence = sequence.transpose(0, 1);
    }
    return {std::move(sequence), std::move(h), std::move(c)};
}

RNN::RNN(Index input_size, Index hidden_size, RecurrentOptions recurrent, RNNActivation activation, TensorOptions options)
    : Recurrent(input_size, hidden_size, activation == RNNActivation::relu ? RecurrentKind::rnn_relu : RecurrentKind::rnn_tanh,
        recurrent, options) {}

GRU::GRU(Index input_size, Index hidden_size, RecurrentOptions recurrent, TensorOptions options)
    : Recurrent(input_size, hidden_size, RecurrentKind::gru, recurrent, options) {}

LSTM::LSTM(Index input_size, Index hidden_size, RecurrentOptions recurrent, TensorOptions options)
    : Recurrent(input_size, hidden_size, RecurrentKind::lstm, recurrent, options) {}

} // namespace dl::nn
