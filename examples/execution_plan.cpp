#include <dl/dl.h>

#include <array>
#include <iostream>
#include <optional>

int main() {
    dl::manual_seed(41);
    const std::array<float, 5> x{-2, -1, 0, 1, 2}, y{-3, -1, 1, 3, 5};
    auto input = dl::from_span<float>(x, {5, 1});
    auto target = dl::from_span<float>(y, {5, 1});
    auto weight = dl::randn({8, 1}, {.requires_grad = true});
    auto bias = dl::zeros({8}, {.requires_grad = true});
    auto head = dl::randn({1, 8}, {.requires_grad = true});
    auto offset = dl::zeros({1}, {.requires_grad = true});

    namespace op = dl::plan::ops;
    dl::plan::Graph graph;
    const auto data = graph.input(dl::tensor_spec(input), "input");
    const auto w = graph.input(dl::tensor_spec(weight), "weight");
    const auto b = graph.input(dl::tensor_spec(bias), "bias");
    const auto h = graph.input(dl::tensor_spec(head), "head");
    const auto c = graph.input(dl::tensor_spec(offset), "offset");
    // 首层单独表达偏置与 ReLU，保留图中的逐元素融合示例。
    auto features = graph.call(op::linear(), {data, w});
    features = graph.call(op::add(), {features, b});
    features = graph.call(op::relu(), {features});
    auto prediction = graph.call(op::linear(), {features, h, c});
    const auto flat = graph.call(op::reshape({-1}), {prediction});

    // 两个输出共享同一份前向存储，求导边仍由各自的 eager 操作建立。
    auto forward = graph.compile({prediction, flat});
    // 同一张图可以保留逐算子执行，两个计划都在调用时重新绑定当前参数。
    auto separate = graph.compile({prediction, flat}, {.fuse_pointwise = false});
    std::cout << "steps " << forward.step_count() << ", unfused steps " << separate.step_count()
        << ", fusions " << forward.fusion_count() << ", handle slots " << forward.slot_count()
        << ", visible storage groups " << forward.storage_lifetimes().size() << '\n';
    for (const auto& buffer : forward.buffer_requirements()) {
        std::cout << "forward buffers: " << buffer.count << " x " << buffer.bytes << " bytes\n";
    }
    // 预留前向尺寸类，训练中仍被反向节点持有的块继续保留，其余块供后续调用复用。
    auto workspace = forward.prepare();

    dl::optim::AdamW optimizer({weight, bias, head, offset}, {.learning_rate = 0.02});
    std::optional<dl::Tensor> last_loss;
    for (int step = 0; step < 40; ++step) {
        last_loss.reset();
        // 普通训练代码共用一个分配作用域，首次创建的梯度和更新状态也借用池中存储。
        last_loss = workspace.with_memory([&] {
            optimizer.zero_grad();
            auto outputs = workspace({input, weight, bias, head, offset});
            auto loss = dl::mse_loss(outputs[0], target);
            loss.backward();
            optimizer.step();
            return loss;
        });
    }
    // 返回的张量在作用域之外仍持有结果，整个训练阶段结束后再读取标量。
    std::cout << "last training loss " << last_loss->item<float>() << '\n';
    last_loss.reset();

    // 调用侧的 NoGrad 选择推理；计划内的参数槽绑定当前已更新的同身份张量。
    dl::NoGradGuard no_grad;
    auto outputs = workspace({input, weight, bias, head, offset});
    for (const auto value : outputs[1].to_vector<float>()) std::cout << value << ' ';
    std::cout << '\n';

    auto memory = workspace.memory_stats(input.device());
    std::cout << "reserved bytes " << memory.reserved_bytes << ", in use " << memory.in_use_bytes << '\n';
    // 回收空闲块；叶梯度和 AdamW 状态仍持有各自的块，CUDA 归还沿原计算 stream 提交。
    outputs.clear();
    workspace.release_unused_memory();
    memory = workspace.memory_stats(input.device());
    std::cout << "retained bytes after release " << memory.reserved_bytes << '\n';
}
