#include <dl/dl.h>

#include <iostream>

int main() {
    const float values[] = {1, 2, 3};
    auto x = dl::from_span<float>(values, {3}, true);
    auto unused = dl::ones({3}, {.requires_grad = true});
    auto square = x * x;
    auto scaled = square * 3.0;
    auto first_seed = dl::ones_like(square), second_seed = dl::full_like(scaled, 2.0);

    // 两个输出共用 square；它既交付自身梯度，也继续把贡献传给另一个请求 x。
    auto gradients = dl::grad({square, scaled}, {square, x, unused, x}, {first_seed, second_seed});
    const char* names[] = {"square", "x", "unused", "x again"};
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        std::cout << names[i] << ':';
        if (gradients[i]) {
            for (const auto value : gradients[i]->to_vector<float>()) std::cout << ' ' << value;
        } else {
            std::cout << " no contribution";
        }
        std::cout << '\n';
    }

    // 另一份前向图可直接把梯度交给优化器；每次更新仍由调用方明确安排。
    auto loss = dl::sum(x * x);
    dl::GradScaler scaler({.initial_scale = 128.0});
    auto scaled_gradient = dl::grad(scaler.scale(loss), {x});
    auto unscaled = scaler.unscale(scaled_gradient);
    if (unscaled.finite) {
        // 独立返回保留此前梯度，适合在同一份结果上比较不同的更新规则。
        auto clipped = dl::optim::clip_gradients_norm(unscaled.gradients, 1.0);
        auto next = dl::optim::sgd(x, *clipped.gradients[0], std::nullopt, {.learning_rate = 0.1});
        std::cout << "gradient norm " << clipped.total_norm.item<float>() << ", updated x:";
        for (const auto value : next.parameter.to_vector<float>()) std::cout << ' ' << value;
        std::cout << '\n';
    }
    scaler.update(unscaled.finite);
}
