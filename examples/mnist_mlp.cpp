#include "mnist_training.h"

int main(int argc, char** argv) {
    return examples::train_mnist(argc, argv, [] {
        auto model = std::make_unique<dl::nn::Sequential>();
        model->append<dl::nn::Flatten>("flatten");
        model->append<dl::nn::Linear>("hidden", 784, 128);
        model->append<dl::nn::ReLU>("activation");
        model->append<dl::nn::Linear>("output", 128, 10);
        return model;
    });
}
