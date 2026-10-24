#include "classification_training.h"
#include "cifar_data.h"
#include "project_models.h"

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: ctorch_cifar_resnet <cifar-10-batches-bin> <output-directory>"
                  << " [--device cpu|index] [--epochs N] [--batch-size N]"
                  << " [--micro-batch-size N] [--precision f32|bf16|f16]"
                  << " [--resume checkpoint] [--stop-after step]\n";
        return 1;
    }
    const std::filesystem::path directory = argv[1];
    return examples::train_classifier(examples::training_options(argc, argv, 128),
        examples::load_cifar10(directory, true), 5000, examples::resnet20,
        [&] { return examples::load_cifar10(directory, false); });
}
