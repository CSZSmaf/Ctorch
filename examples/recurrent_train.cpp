#include <dl/dl.h>

#include <array>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    auto device = dl::Device::cpu();
    int checkpoint_argument = 1;
    if (argc > 2 && std::string_view(argv[1]) == "--cuda") {
        device = dl::Device::cuda(std::stoi(argv[2]));
        checkpoint_argument = 3;
    }
    dl::manual_seed(31);
    constexpr dl::Index length = 4, batch = 8;
    std::array<float, length * batch> inputs{}, targets{};
    // 目标是每条序列的前缀和，模型需要从隐藏状态恢复此前的输入。
    for (dl::Index sample = 0; sample < batch; ++sample) {
        float total = 0;
        for (dl::Index time = 0; time < length; ++time) {
            const auto index = static_cast<std::size_t>(time * batch + sample);
            inputs[index] = static_cast<float>((sample * 3 + time * 5) % 7 - 3) * 0.2f;
            total += inputs[index];
            targets[index] = total;
        }
    }
    auto input = dl::from_span<float>(inputs, {length, batch, 1}, false, device);
    auto target = dl::from_span<float>(targets, {length, batch, 1}, false, device);
    dl::nn::Sequential model;
    auto memory = model.append<dl::nn::GRU>("memory", 1, 8);
    auto readout = model.append<dl::nn::Linear>("readout", 8, 1);
    // 参数先迁移到计算设备，优化器随后在同一设备创建各参数的状态。
    model.to(device);
    dl::optim::Adam optimizer(std::vector<dl::optim::ParameterGroup>{
        {memory->parameters(), 0.02, 0}, {readout->parameters(), 0.03, 0}});
    dl::optim::CosineAnnealingLR scheduler(optimizer, 200, 0.001);
    const std::array schedules{dl::io::NamedScheduler{"learning_rate", scheduler}};
    dl::io::TrainingProgress progress;
    if (argc > checkpoint_argument) {
        progress = dl::io::restore_checkpoint(dl::io::load_checkpoint(argv[checkpoint_argument]), model, optimizer, nullptr, {}, schedules);
    }
    for (dl::Index step = progress.step + 1; step <= 200; ++step) {
        optimizer.zero_grad();
        auto loss = dl::mse_loss(model(input), target);
        loss.backward();
        dl::optim::clip_grad_norm(optimizer.parameters(), 1.0);
        optimizer.step();
        scheduler.step();
        if (step == 1 || step % 50 == 0) std::cout << "step " << step << ", loss " << loss.item<float>() << '\n';
    }
    dl::io::save_checkpoint("recurrent.train.ct", dl::io::capture_checkpoint(model, optimizer, {.step = 200}, nullptr, {}, schedules));
    model.eval();
    dl::NoGradGuard no_grad;
    std::cout << "final loss " << dl::mse_loss(model(input), target).item<float>() << '\n';
}
