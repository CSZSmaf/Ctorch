#include <dl/dl.h>

#include <iostream>

int main() {
    dl::manual_seed(43);
    const dl::TensorOptions trainable{.requires_grad = true};
    auto sequence = dl::randn({4, 2, 3}, trainable);
    auto hidden = dl::zeros({2, 5}, trainable);
    auto cell = dl::zeros({2, 5}, trainable);
    auto weight_ih = dl::full({20, 3}, 0.1, trainable);
    auto weight_hh = dl::full({20, 5}, 0.05, trainable);
    auto bias_ih = dl::zeros({20}, trainable);
    auto bias_hh = dl::zeros({20}, trainable);

    namespace op = dl::plan::ops;
    dl::plan::Graph graph;
    const auto x = graph.input(dl::tensor_spec(sequence), "sequence");
    auto h = graph.input(dl::tensor_spec(hidden), "hidden");
    auto c = graph.input(dl::tensor_spec(cell), "cell");
    const auto wi = graph.input(dl::tensor_spec(weight_ih), "weight_ih");
    const auto wh = graph.input(dl::tensor_spec(weight_hh), "weight_hh");
    const auto bi = graph.input(dl::tensor_spec(bias_ih), "bias_ih");
    const auto bh = graph.input(dl::tensor_spec(bias_hh), "bias_hh");

    // unbind 一次创建时间步视图；每个 LSTM 调用同时生成 hidden 和 cell。
    const auto steps = graph.call(op::unbind(0), {x});
    for (const auto step : steps) {
        auto next = graph.call(op::lstm_cell(), {step, h, c, wi, wh, bi, bh});
        h = next[0];
        c = next[1];
    }
    auto forward = graph.compile({h, c});
    auto outputs = forward({sequence, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh});

    // 两个输出共同进入一个损失，已有求导引擎处理共享祖先和跨时间步参数累计。
    auto loss = dl::mean(outputs[0] * outputs[0]) + dl::mean(outputs[1] * outputs[1]);
    loss.backward();
    std::cout << "steps " << forward.step_count() << ", handle slots " << forward.slot_count()
        << ", loss " << loss.item<float>()
        << ", weight gradient mean " << dl::mean(dl::abs(*weight_ih.grad())).item<float>() << '\n';
}
