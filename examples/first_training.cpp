#include <dl/dl.h>

#include <array>
#include <iostream>

int main() {
    const std::array<float, 5> x{-2, -1, 0, 1, 2}, y{-3, -1, 1, 3, 5};
    auto input = dl::from_span<float>(x, {5, 1});
    auto target = dl::from_span<float>(y, {5, 1});

    // 显式的零初值让每一步都能与教程中的线性回归演示对应。
    dl::nn::Linear model(dl::zeros({1, 1}, {.requires_grad = true}),
                         dl::zeros({1}, {.requires_grad = true}));
    dl::optim::SGD optimizer(model.parameters(), {.learning_rate = 0.1});
    for (int step = 0; step < 100; ++step) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(model(input), target);
        loss.backward();
        optimizer.step();
        if (step == 0) std::cout << "initial loss " << loss.item<float>() << '\n';
    }

    // 评估使用当前参数；NoGradGuard 让本次前向只保留数值结果。
    model.eval();
    dl::NoGradGuard no_grad;
    std::cout << "weight " << model.weight().item<float>()
              << ", bias " << model.bias()->item<float>() << '\n';
    std::cout << "final loss " << dl::mse_loss(model(input), target).item<float>() << '\n';
}
