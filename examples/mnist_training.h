#pragma once

#include "classification_training.h"

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
    return train_classifier(options, std::move(dataset), 5000, std::move(build_model), [&] {
        return dl::data::load_mnist(directory / "t10k-images-idx3-ubyte", directory / "t10k-labels-idx1-ubyte");
    });
}

} // namespace examples
