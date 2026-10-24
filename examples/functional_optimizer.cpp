#include <dl/dl.h>

#include <iostream>

void print(const char* label, const dl::Tensor& parameter) {
    std::cout << label;
    for (const auto value : parameter.to_vector<float>()) std::cout << ' ' << value;
    std::cout << '\n';
}

// 训练过程只接收一步更新函数，状态由下面各算法的调用方直接持有。
template <class Update>
void fit(const char* label, const dl::Tensor& input, const dl::Tensor& target, Update update) {
    auto weight = dl::zeros({2, 1}, {.requires_grad = true});
    for (int step = 0; step < 16; ++step) {
        auto loss = dl::mse_loss(dl::matmul(input, weight), target);
        auto gradients = dl::grad(loss, {weight});
        weight = update(weight, *gradients[0]);
    }
    print(label, weight);
}

int main() {
    const float samples[] = {1, 0, 0, 1, 1, 1, 2, -1};
    const float targets[] = {2, -1, 1, 5};
    auto input = dl::from_span<float>(samples, {4, 2});
    auto target = dl::from_span<float>(targets, {4, 1});

    std::optional<dl::Tensor> velocity;
    const dl::optim::SGDOptions sgd{.learning_rate = 0.08, .momentum = 0.9, .nesterov = true};
    fit("SGD weight", input, target, [&](const dl::Tensor& parameter, const dl::Tensor& gradient) {
        auto next = dl::optim::sgd(parameter, gradient, velocity, sgd);
        // 替换调用方的参数和状态；新叶的梯度槽为空，不保留上一步求导图。
        velocity = std::move(next.velocity);
        return std::move(next.parameter);
    });

    std::optional<dl::optim::AdamMoments> moments;
    fit("Adam weight", input, target, [&](const dl::Tensor& parameter, const dl::Tensor& gradient) {
        auto next = dl::optim::adam(parameter, gradient, moments, {.learning_rate = 0.05, .weight_decay = 0.01});
        moments = std::move(next.moments);
        return std::move(next.parameter);
    });

    // 原地形式沿用同一参数和统计缓冲；直接返回梯度的训练过程也可采用这一形式。
    std::optional<dl::optim::RMSpropBuffers> buffers;
    fit("RMSprop weight", input, target, [&](dl::Tensor parameter, const dl::Tensor& gradient) {
        dl::optim::rmsprop_(parameter, gradient, buffers,
            {.learning_rate = 0.02, .momentum = 0.9, .centered = true});
        return parameter;
    });
    std::cout << "RMSprop updates " << buffers->step << '\n';

    std::optional<dl::optim::AdagradAccumulator> accumulator;
    fit("Adagrad weight", input, target, [&](const dl::Tensor& parameter, const dl::Tensor& gradient) {
        auto next = dl::optim::adagrad(parameter, gradient, accumulator,
            {.learning_rate = 0.1, .learning_rate_decay = 0.02, .initial_accumulator = 0.1});
        accumulator = std::move(next.accumulator);
        return std::move(next.parameter);
    });
    std::cout << "Adagrad updates " << accumulator->step << '\n';

    auto weight = dl::zeros({2, 1}, {.requires_grad = true});
    auto first = dl::zeros_like(weight), second = dl::zeros_like(weight);
    const dl::Index initial_step = 0;
    auto counter = dl::from_span<dl::Index>(std::span<const dl::Index>(&initial_step, 1), {});

    // 更新图只描述一步；参数、梯度、两个矩估计和计数在每次调用时重新绑定。
    dl::plan::Graph graph;
    const auto spec = dl::tensor_spec(weight);
    const auto p = graph.input(spec, "weight"), g = graph.input(spec, "gradient");
    const auto m = graph.input(spec, "first"), v = graph.input(spec, "second");
    const auto t = graph.input(dl::tensor_spec(counter), "step");
    const auto values = graph.call(dl::plan::ops::adamw({.learning_rate = 0.05, .weight_decay = 0.0}),
        {p, g, m, v, t});
    const auto update = graph.compile(values);

    for (int step = 0; step < 16; ++step) {
        auto loss = dl::mse_loss(dl::matmul(input, weight), target);
        auto gradients = dl::grad(loss, {weight});
        auto next = update({weight, *gradients[0], first, second, counter});
        weight = std::move(next[0]);
        first = std::move(next[1]);
        second = std::move(next[2]);
        counter = std::move(next[3]);
    }
    print("AdamW weight", weight);
    std::cout << "AdamW updates " << *counter.data<dl::Index>() << '\n';
}
