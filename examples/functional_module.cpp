#include <dl/dl.h>

#include <algorithm>
#include <array>
#include <iostream>

int main() {
    const float weights[] = {0.2f, -0.1f, 0.3f, 0.4f};
    auto shared = dl::from_span<float>(weights, {2, 2}, true);
    dl::nn::Sequential model;
    model.append<dl::nn::Linear>("input", shared);
    model.append<dl::nn::BatchNorm1d>("norm", 2);
    model.append<dl::nn::SiLU>("activation");
    model.append<dl::nn::Dropout>("dropout", 0.2);
    model.append<dl::nn::Linear>("output", shared);

    // 两个 Linear 共用一个参数槽；独立状态从当前模块数值建立一次。
    dl::nn::FunctionalModule functional(model);
    auto state = functional.state();
    std::vector<std::optional<dl::optim::AdamMoments>> moments(state.parameters.size());
    dl::RandomKey next_key{73};
    dl::GradScaler scaler({.initial_scale = 128.0, .growth_interval = 4});
    dl::Index updates = 0;
    dl::optim::AdamWOptions options{.learning_rate = 0.02, .weight_decay = 0.0};
    const float samples[] = {1, 0, 0, 1, 1, 1, 2, -1};
    const float targets[] = {0.5f, 0, 0, 0.5f, 0.5f, 0.5f, 1, -0.5f};
    auto input = dl::from_span<float>(samples, {4, 2});
    auto target = dl::from_span<float>(targets, {4, 2});

    // 模型结构与训练模式固定在计划中；每次前向重新绑定当前的参数、buffer 和 key。
    std::vector<dl::Tensor> bindings{input};
    bindings.insert(bindings.end(), state.parameters.begin(), state.parameters.end());
    bindings.insert(bindings.end(), state.buffers.begin(), state.buffers.end());
    bindings.push_back(dl::key_tensor(next_key));
    dl::plan::Graph graph;
    std::vector<dl::plan::Value> slots;
    for (const auto& value : bindings) slots.push_back(graph.input(dl::tensor_spec(value)));
    auto operation = dl::plan::ops::functional_module(functional, state.training,
        [](std::span<const dl::TensorSpec> inputs) {
            return std::vector<dl::plan::OutputSpec>{
                {dl::contiguous_spec(inputs[0].shape, dl::DType::f32, inputs[0].device)}};
        }, [&model](std::span<const dl::Tensor> inputs) {
            return std::vector<dl::Tensor>{model(inputs[0])};
        });
    const auto values = graph.call(std::move(operation), slots);
    auto workspace = graph.compile(values).prepare();

    auto train_until = [&](std::uint64_t end) {
        std::optional<dl::Tensor> last_loss;
        while (next_key.stream < end) {
            // 前向、梯度值和后继参数共用池；长期状态继续由 state/moments 自己持有。
            last_loss = workspace.with_memory([&] {
                std::copy(state.parameters.begin(), state.parameters.end(), bindings.begin() + 1);
                std::copy(state.buffers.begin(), state.buffers.end(), bindings.begin() + 1 + state.parameters.size());
                bindings.back() = dl::key_tensor(next_key);
                auto result = workspace(bindings);
                ++next_key.stream;
                auto loss = dl::mse_loss(result.back(), target);
                auto gradients = dl::grad(scaler.scale(loss), state.parameters);
                const bool finite = scaler.unscale_(gradients);
                if (finite) {
                    dl::optim::clip_gradients_norm_(gradients, 1.0);
                    for (std::size_t i = 0; i < state.parameters.size(); ++i) {
                        auto& parameter = state.parameters[i];
                        if (gradients[i]) {
                            auto next = dl::optim::adamw(parameter, *gradients[i], moments[i], options);
                            parameter = std::move(next.parameter);
                            moments[i] = std::move(next.moments);
                        }
                    }
                    ++updates;
                }
                scaler.update(finite);
                // buffer 与随机位置按前向推进，优化器计数只在有限梯度更新后推进。
                std::move(result.begin(), result.begin() + state.buffers.size(), state.buffers.begin());
                return loss.detach();
            });
        }
        // 两个训练段各自在完成后读取一次最后的损失。
        std::cout << "forward " << next_key.stream << ", updates " << updates
            << ", loss " << last_loss->item<float>() << '\n';
    };
    train_until(6);

    // 快照保存外部训练分支；恢复交付新的参数、buffer、动量和随机 key。
    auto checkpoint = dl::io::capture_checkpoint(functional, state,
        dl::optim::AdamWState{options, moments, {}}, next_key, {.step = updates}, nullptr, {}, &scaler);
    auto restored = dl::io::restore_checkpoint(checkpoint, functional, dl::Device::cpu(), nullptr, {}, &scaler);
    state = std::move(restored.state);
    next_key = restored.random_key;
    updates = restored.progress.step;
    auto optimizer = std::get<dl::optim::AdamWState>(std::move(restored.optimizer));
    options = optimizer.options;
    moments = std::move(optimizer.moments);
    train_until(12);

    // 模式由状态表选择；普通 Module 的配置和原始参数继续保留。
    std::fill(state.training.begin(), state.training.end(), std::uint8_t{0});
    dl::NoGradGuard no_grad;
    // 这些层共用原权重，内层投影、外层模型和独立参考层各自选择自己的绑定。
    dl::nn::Linear projection(shared), reference(shared);
    dl::nn::FunctionalModule inner(projection);
    auto inner_state = inner.state();
    inner_state.parameters[0] = dl::full({2, 2}, 0.5);
    auto evaluated = functional.call(state, next_key, [&] {
        auto result = inner.call(inner_state, dl::fold_in(next_key, 1), [&] {
            auto prediction = model(input);
            return std::array{prediction, projection(prediction), reference(input)};
        });
        return std::move(result.output);
    });
    std::cout << "eval loss " << dl::mse_loss(evaluated.output[0], target).item<float>() << '\n';
    std::cout << "nested projection " << evaluated.output[1].data<float>()[0]
        << ", original projection " << evaluated.output[2].data<float>()[0] << '\n';
    for (std::size_t i = 0; i < functional.parameter_names().size(); ++i) {
        std::cout << "parameter " << i << ": " << functional.parameter_names()[i] << '\n';
    }
    std::cout << "original shared weight " << shared.data<float>()[0]
        << ", updated weight " << state.parameters[0].data<float>()[0] << '\n';
}
