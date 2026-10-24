#include <dl/dl.h>

#include <array>
#include <iostream>
#include <optional>

int main() {
    dl::manual_seed(41);
    const std::array<float, 5> x{-2, -1, 0, 1, 2}, y{-3, -1, 1, 3, 5};
    auto input = dl::from_span<float>(x, {5, 1});
    auto target = dl::from_span<float>(y, {5, 1});
    auto weight = dl::randn({1, 8}, {.requires_grad = true});
    auto bias = dl::zeros({8}, {.requires_grad = true});
    auto head = dl::randn({8, 1}, {.requires_grad = true});
    auto offset = dl::zeros({1}, {.requires_grad = true});
    dl::optim::AdamW optimizer({weight, bias, head, offset}, {.learning_rate = 0.02});

    // 子计划将数据和块内参数全部放入输入槽，编译时确定融合及计算精度。
    namespace op = dl::plan::ops;
    dl::plan::Graph graph;
    const auto x_slot = graph.input(dl::tensor_spec(input), "input");
    const auto w_slot = graph.input(dl::tensor_spec(weight), "weight");
    const auto b_slot = graph.input(dl::tensor_spec(bias), "bias");
    auto hidden = graph.call(op::matmul(), {x_slot, w_slot});
    hidden = graph.call(op::add(), {hidden, b_slot});
    hidden = graph.call(op::relu(), {hidden});
    hidden = graph.call(op::dropout(0.1), {hidden});
    const auto block = graph.compile({hidden});
    auto workspace = block.prepare();

    std::optional<dl::Tensor> last_loss;
    for (int step = 0; step < 40; ++step) {
        last_loss.reset();
        // 分配作用域覆盖重算、梯度和更新；随机流重放在重算入口内部完成。
        last_loss = workspace.with_memory([&] {
            optimizer.zero_grad();
            auto features = block.checkpoint({input, weight, bias});
            auto loss = dl::mse_loss(dl::matmul(features, head) + offset, target);
            loss.backward();
            optimizer.step();
            return loss;
        });
    }
    std::cout << "last training loss " << last_loss->item<float>() << '\n';

    // 显式 key 已描述随机序列，可省去默认 Generator 快照；grad 仍返回独立梯度值。
    auto features = dl::checkpoint({input, weight, bias}, [key = dl::RandomKey{41, 7}](auto values) {
        return dl::dropout(dl::relu(dl::matmul(values[0], values[1]) + values[2]), 0.1, key, true);
    }, {.preserve_rng_state = false});
    auto gradients = dl::grad(dl::sum(features), {weight, bias});
    std::cout << "weight gradient elements " << gradients[0]->numel() << '\n';
}
