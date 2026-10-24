#include <dl/dl.h>

#include <array>
#include <iostream>

class Regressor final : public dl::nn::UnaryModule {
public:
    Regressor() : features(add_module<dl::nn::Linear>("features", 1, 8)),
        head(add_module<dl::nn::Linear>("head", 8, 1)) {}

    dl::Tensor forward(const dl::Tensor& input) override {
        return (*head)(dl::tanh((*features)(input)));
    }

    std::shared_ptr<dl::nn::Linear> features, head;
};

float fit(Regressor& model, dl::optim::Optimizer& optimizer,
    const dl::Tensor& input, const dl::Tensor& target, dl::Index steps) {
    float last_loss = 0;
    for (dl::Index step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(model(input), target);
        loss.backward();
        optimizer.step();
        last_loss = loss.item<float>();
    }
    return last_loss;
}

int main() {
    dl::manual_seed(37);
    const std::array<float, 5> x{-2, -1, 0, 1, 2}, y{-3, -1, 1, 3, 5};
    auto input = dl::from_span<float>(x, {5, 1});
    auto target = dl::from_span<float>(y, {5, 1});
    Regressor model;
    model.features->requires_grad_(false);

    // 优化器从开始就登记完整参数集合，冻结期间特征层保留空状态槽。
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = 0.02});
    constexpr dl::Index head_steps = 20, joint_steps = 20;
    const auto head_loss = fit(model, optimizer, input, target, head_steps);
    auto checkpoint = dl::io::capture_checkpoint(model, optimizer, {.step = head_steps});
    std::cout << "head loss " << head_loss
        << ", trainable tensors " << model.trainable_parameters().size() << '\n';

    // 新对象恢复权重、动量、随机流与冻结状态；本例在内存中交付快照。
    Regressor resumed;
    dl::optim::AdamW resumed_optimizer(resumed.parameters(), {.learning_rate = 0.02});
    const auto progress = dl::io::restore_checkpoint(checkpoint, resumed, resumed_optimizer);
    std::cout << "resumed step " << progress.step
        << ", trainable tensors " << resumed.trainable_parameters().size() << '\n';

    // 解冻后复用同一优化器，特征层从本阶段第一次梯度开始建立矩估计。
    resumed.features->requires_grad_(true);
    const auto joint_loss = fit(resumed, resumed_optimizer, input, target, joint_steps);
    std::cout << "joint loss " << joint_loss
        << ", trainable tensors " << resumed.trainable_parameters().size() << '\n';
}
