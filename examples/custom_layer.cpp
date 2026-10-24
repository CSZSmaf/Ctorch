#include <dl/dl.h>

#include <array>
#include <iostream>

/// SiLU 由乘法和 sigmoid 组合，已有 VJP 沿这两项运算完成链式求导。
class SiLU : public dl::nn::UnaryModule {
public:
    dl::Tensor forward(const dl::Tensor& input) override { return input * dl::sigmoid(input); }
};

int main() {
    dl::manual_seed(23);
    dl::nn::Sequential model;
    model.append<dl::nn::Linear>("input", 1, 8);
    model.append<SiLU>("activation");
    model.append<dl::nn::Linear>("output", 8, 1);
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = 0.01, .weight_decay = 0.0});
    const std::array<float, 5> xs{-2, -1, 0, 1, 2}, ys{4, 1, 0, 1, 4};
    auto input = dl::from_span<float>(xs, {5, 1});
    auto target = dl::from_span<float>(ys, {5, 1});
    for (int step = 0; step < 200; ++step) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(model(input), target);
        loss.backward();
        optimizer.step();
        if ((step + 1) % 20 == 0) {
            std::cout << "step " << step + 1 << ", loss " << loss.item<float>() << '\n';
        }
    }
    dl::io::save("custom_layer.ct", model.state_dict());
}
