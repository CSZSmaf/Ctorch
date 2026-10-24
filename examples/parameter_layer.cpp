#include <dl/dl.h>

#include <iostream>

/// 参数在构造时登记，前向通过 bound 读取当前调用的参数绑定。
class Scale : public dl::nn::UnaryModule {
public:
    Scale() : weight_(register_parameter("weight", dl::full({}, 2.0, {.requires_grad = true}))) {}
    dl::Tensor forward(const dl::Tensor& input) override { return input * bound(weight_); }

private:
    dl::Tensor weight_;
};

int main() {
    Scale model;
    auto input = dl::full({}, 3.0);
    std::cout << dl::summary(model) << '\n';
    std::cout << "module output " << model(input).item<float>() << '\n';

    // 外部状态拥有独立的参数存储，调用作用域结束后普通模型仍使用自身参数。
    dl::nn::FunctionalModule functional(model);
    auto state = functional.state();
    state.parameters[0] = dl::full({}, 5.0, {.requires_grad = true});
    auto result = functional.call(state, dl::RandomKey{7}, [&] { return model(input); });
    std::cout << "functional output " << result.output.item<float>() << '\n';
    std::cout << "module output " << model(input).item<float>() << '\n';

    // 优化器从 Module 收集登记的叶参数，自定义层沿普通训练路径更新。
    dl::optim::SGD optimizer(model.parameters(), {.learning_rate = 0.05});
    auto target = dl::full({}, 9.0);
    for (int step = 0; step < 10; ++step) {
        optimizer.zero_grad();
        dl::mse_loss(model(input), target).backward();
        optimizer.step();
    }
    std::cout << "trained output " << model(input).item<float>() << '\n';
}
