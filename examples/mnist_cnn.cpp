#include "mnist_training.h"

int main(int argc, char** argv) {
    return examples::train_mnist(argc, argv, [] {
        auto model = std::make_unique<dl::nn::Sequential>();
        model->append<dl::nn::Conv2d>("conv1", 1, 8, dl::Size2{3}, dl::Conv2dOptions{.padding = 1});
        model->append<dl::nn::BatchNorm2d>("norm1", 8);
        model->append<dl::nn::ReLU>("relu1");
        model->append<dl::nn::MaxPool2d>("pool1");
        model->append<dl::nn::Conv2d>("conv2", 8, 16, dl::Size2{3}, dl::Conv2dOptions{.padding = 1});
        model->append<dl::nn::BatchNorm2d>("norm2", 16);
        model->append<dl::nn::ReLU>("relu2");
        model->append<dl::nn::MaxPool2d>("pool2");
        model->append<dl::nn::Flatten>("flatten");
        model->append<dl::nn::Linear>("output", 16 * 7 * 7, 10);
        return model;
    });
}
