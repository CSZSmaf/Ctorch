#pragma once

#include "training.h"

#include <memory>

namespace examples {

/// MNIST 官方训练集的末 5,000 个样本用于验证，测试集只在最佳权重选定后读取。
template <class BuildModel>
int train_mnist(int argc, char** argv, BuildModel build_model) {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: " << argv[0] << " <mnist-directory> <output-directory>"
                  << " [--device cpu|index] [--epochs N] [--batch-size N]"
                  << " [--micro-batch-size N] [--clip-grad-norm value]"
                  << " [--precision f32|bf16|f16] [--resume checkpoint] [--stop-after step]\n";
        return 1;
    }
    auto options = training_options(argc, argv);
    const std::filesystem::path directory = argv[1];
    auto dataset = dl::data::load_mnist(directory / "train-images-idx3-ubyte", directory / "train-labels-idx1-ubyte");
    const auto training_size = dataset.size() - 5000;
    dl::Generator sampler(7), evaluation_sampler(0);
    dl::data::DataLoader source({dataset.input.narrow(0, 0, training_size), dataset.target.narrow(0, 0, training_size)},
        {.batch_size = options.batch_size}, sampler);
    dl::data::PrefetchLoader loader(source, options.device);
    dl::data::DataLoader validation({dataset.input.narrow(0, training_size, 5000), dataset.target.narrow(0, training_size, 5000)},
        {.batch_size = 256, .shuffle = false}, evaluation_sampler);
    dl::manual_seed(42);
    auto model = build_model();
    model->to(options.device);
    dl::optim::AdamW optimizer(model->parameters(), {.learning_rate = 0.001});
    TrainingState state;
    auto progress = begin_training(options, *model, optimizer, loader, validation, state);
    while (progress.epoch < options.epochs) {
        while (auto batch = loader.next()) {
            const bool updated = train_batch(options, *model, *batch, optimizer, state, batch->input.size(0));
            if (updated) ++progress.step;
            if (progress.step >= options.stop_after) {
                save_training(options, *model, optimizer, progress, loader, state);
                std::cout << "saved step " << progress.step << std::endl;
                return 0;
            }
        }
        ++progress.epoch;
        record_validation(options, *model, progress, state,
            evaluate(*model, validation, options.device, options.precision, options.micro_batch_size));
        loader.reset();
        save_training(options, *model, optimizer, progress, loader, state);
    }
    dl::data::DataLoader test(dl::data::load_mnist(directory / "t10k-images-idx3-ubyte", directory / "t10k-labels-idx1-ubyte"),
        {.batch_size = 256, .shuffle = false}, evaluation_sampler);
    finish_training(options, *model, test, state, progress);
    return 0;
}

} // namespace examples
