#include <dl/dl.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// 每个样本有两个坐标。标签由 x + y > 0.1 决定。
dl::data::TensorDataset grid(dl::Index side) {
    std::vector<float> points;
    std::vector<dl::Index> labels;
    for (dl::Index row = 0; row < side; ++row) {
        for (dl::Index column = 0; column < side; ++column) {
            const auto x = static_cast<float>(-1.0 + (row + 0.5) * 2.0 / side);
            const auto y = static_cast<float>(-1.0 + (column + 0.5) * 2.0 / side);
            points.insert(points.end(), {x, y});
            labels.push_back(x + y > 0.1f ? 1 : 0);
        }
    }
    return {dl::from_span<float>(points, {side * side, 2}),
            dl::from_span<dl::Index>(labels, {side * side})};
}

std::unique_ptr<dl::nn::Sequential> classifier() {
    auto model = std::make_unique<dl::nn::Sequential>();
    model->append<dl::nn::Linear>("hidden", 2, 8);
    model->append<dl::nn::ReLU>("activation");
    model->append<dl::nn::Linear>("output", 8, 2);
    return model;
}

struct Metrics {
    double loss;
    double accuracy;
};

// 最后一个批次可能较小，因此损失按样本数加权。
Metrics evaluate(dl::nn::Sequential& model, dl::data::DataLoader& loader) {
    model.eval();
    dl::NoGradGuard no_grad;
    loader.reset();
    double loss_sum = 0;
    dl::Index count = 0, correct = 0;
    while (auto batch = loader.next()) {
        auto logits = model(batch->input);
        const auto size = batch->target.numel();
        loss_sum += dl::cross_entropy(logits, batch->target).item<float>() * size;
        auto matches = dl::eq(dl::argmax(logits, 1), batch->target);
        correct += static_cast<dl::Index>(dl::sum(matches.astype(dl::DType::f32)).item<float>());
        count += size;
    }
    return {loss_sum / count, static_cast<double>(correct) / count};
}

int main(int argc, char** argv) {
    const auto path = argc > 1 ? argv[1] : "training-basics.ct";
    dl::manual_seed(11);
    dl::Generator sampler{19}, evaluation{0};
    // 两个网格使用不同的坐标；验证样本不参加参数更新。
    dl::data::DataLoader training(grid(8), {.batch_size = 16}, sampler);
    dl::data::DataLoader validation(grid(5), {.batch_size = 16, .shuffle = false}, evaluation);
    auto model = classifier();
    dl::optim::Adam optimizer(model->parameters(), {.learning_rate = 0.03});
    dl::optim::StepLR scheduler(optimizer, 50, 0.5);
    dl::Index step = 0;
    std::cout << std::fixed << std::setprecision(6);

    for (dl::Index epoch = 0; epoch < 20; ++epoch) {
        model->train();
        training.reset();
        while (auto batch = training.next()) {
            optimizer.zero_grad();
            auto loss = dl::cross_entropy((*model)(batch->input), batch->target);
            loss.backward();
            optimizer.step();
            scheduler.step();
            ++step;
        }
        if ((epoch + 1) % 5 == 0) {
            const auto metrics = evaluate(*model, validation);
            std::cout << "epoch " << epoch + 1 << ", loss " << metrics.loss
                      << ", accuracy " << metrics.accuracy << '\n';
        }
    }

    // 保存参数、优化器、采样位置与学习率进度。epoch 指向下一轮。
    std::array schedules{dl::io::NamedScheduler{"learning_rate", scheduler}};
    auto checkpoint = dl::io::capture_checkpoint(
        *model, optimizer, {.epoch = 20, .step = step}, &training, {}, schedules);
    dl::io::save_checkpoint(path, checkpoint);

    auto restored = classifier();
    dl::optim::Adam resumed(restored->parameters(), {.learning_rate = 0.03});
    dl::optim::StepLR resumed_scheduler(resumed, 50, 0.5);
    std::array resumed_schedules{dl::io::NamedScheduler{"learning_rate", resumed_scheduler}};
    const auto progress = dl::io::restore_checkpoint(
        dl::io::load_checkpoint(path), *restored, resumed, &training, {}, resumed_schedules);
    const auto metrics = evaluate(*restored, validation);
    std::cout << "restored step " << progress.step << ", loss " << metrics.loss
              << ", accuracy " << metrics.accuracy << '\n';
}
