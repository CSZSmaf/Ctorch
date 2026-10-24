#include <dl/dl.h>

#include <iostream>

void print(const char* label, const dl::Tensor& tensor) {
    std::cout << label;
    for (const auto value : tensor.to_vector<float>()) std::cout << ' ' << value;
    std::cout << '\n';
}

int main() {
    const dl::RandomKey root{47};
    const auto [noise_key, mask_key] = dl::split(root);
    auto noise = dl::randn(noise_key, {2, 4});
    auto input = dl::ones({2, 4}, {.requires_grad = true});
    auto sampled = dl::dropout(input, 0.25, mask_key);
    print("normal", noise);
    print("dropout", sampled);

    // 图内分流将不同随机用途绑定到独立结果，key 本身始终是显式 CPU 输入。
    namespace op = dl::plan::ops;
    dl::plan::Graph graph;
    const auto data = graph.input(dl::tensor_spec(input), "input");
    const auto key = graph.input(dl::contiguous_spec({2}, dl::DType::i64), "step_key");
    const auto streams = graph.call(op::split_key(), {key});
    const auto random = graph.call(op::randn({2, 4}), {streams[0]});
    const auto dropped = graph.call(op::dropout_with_key(0.25), {data, streams[1]});
    const auto output = graph.call(op::add(), {dropped, random});
    const auto forward = graph.compile({output});

    for (std::uint64_t step = 0; step < 2; ++step) {
        input.zero_grad();
        auto step_key = dl::key_tensor(dl::fold_in(root, step));
        auto result = forward({input, step_key})[0];
        dl::mean(result * result).backward();
        print("planned output", result);
        print("input gradient", *input.grad());
    }

    // 恢复 root 和下一次前向的步号即可重建相应 key，不依赖其他随机调用的数量。
    const auto resumed_key = dl::fold_in(root, 2);
    print("next normal", dl::randn(dl::split(resumed_key)[0], {2, 4}));
}
