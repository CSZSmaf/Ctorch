#include <dl/dl.h>

#include <array>
#include <iostream>

dl::plan::ExecutionPlan make_forward(const dl::Tensor& input, const dl::BatchNormState& state,
    const dl::Tensor& weight, const dl::Tensor& bias, bool training) {
    dl::plan::Graph graph;
    const auto x = graph.input(dl::tensor_spec(input), "input");
    const auto center = graph.input(dl::tensor_spec(state.mean), "running_mean");
    const auto variance = graph.input(dl::tensor_spec(state.variance), "running_variance");
    const auto w = graph.input(dl::tensor_spec(weight), "weight");
    const auto b = graph.input(dl::tensor_spec(bias), "bias");
    auto outputs = graph.call(dl::plan::ops::functional_batch_norm({}, training), {x, center, variance, w, b});
    return graph.compile(outputs);
}

int main() {
    const std::array<float, 12> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto input = dl::from_span<float>(values, {6, 2});
    auto target = dl::ones({6, 2});
    auto weight = dl::ones({2}, {.requires_grad = true});
    auto bias = dl::zeros({2}, {.requires_grad = true});
    const dl::BatchNormState initial{dl::zeros({2}), dl::ones({2})};
    auto state = initial;
    dl::optim::AdamW optimizer({weight, bias}, {.learning_rate = 0.02});

    auto update_parameters = [&](const dl::Tensor& output) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(output, target);
        loss.backward();
        optimizer.step();
        return loss.item<float>();
    };

    // 新状态显式接入下一批；initial 继续持有第一次调用之前的独立值。
    for (int step = 0; step < 3; ++step) {
        auto result = dl::batch_norm(input, state, weight, bias);
        std::cout << "functional loss " << update_parameters(result.output) << '\n';
        state = std::move(*result.state);
    }
    std::cout << "initial mean " << initial.mean.to_vector<float>()[0]
        << ", current mean " << state.mean.to_vector<float>()[0] << '\n';

    // 固定计划一次交付输出和两个新状态，下一批显式绑定这组后继状态。
    auto train_forward = make_forward(input, state, weight, bias, true);
    for (int step = 0; step < 3; ++step) {
        auto outputs = train_forward({input, state.mean, state.variance, weight, bias});
        std::cout << "planned loss " << update_parameters(outputs[0]) << '\n';
        state = {std::move(outputs[1]), std::move(outputs[2])};
    }

    auto eval_forward = make_forward(input, state, weight, bias, false);
    dl::NoGradGuard no_grad;
    auto output = eval_forward({input, state.mean, state.variance, weight, bias})[0];
    std::cout << "eval loss " << dl::mse_loss(output, target).item<float>() << '\n';
}
