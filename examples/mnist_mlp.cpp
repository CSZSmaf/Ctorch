#include "mnist_training.h"
#include "project_models.h"

int main(int argc, char** argv) {
    return examples::train_mnist(argc, argv, examples::mnist_mlp);
}
