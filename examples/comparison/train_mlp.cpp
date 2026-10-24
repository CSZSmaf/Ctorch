#include "../project_models.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: ctorch_compare_training <mnist-directory> <output-directory> [steps]\n";
        return 1;
    }
    const std::filesystem::path directory = argv[1], output = argv[2];
    const dl::Index steps = argc == 4 ? std::stoll(argv[3]) : 200, batch_size = 64, training_size = 55000;
    std::filesystem::create_directories(output);
    auto data = dl::data::load_mnist(directory / "train-images-idx3-ubyte", directory / "train-labels-idx1-ubyte");
    std::vector<dl::Index> permutation(training_size), order(static_cast<std::size_t>(steps * batch_size));
    std::iota(permutation.begin(), permutation.end(), dl::Index{0});
    dl::Generator sampler(7);
    sampler.shuffle(permutation);
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = permutation[i % permutation.size()];
    auto indices = dl::from_span<dl::Index>(order, {steps, batch_size});
    dl::manual_seed(42);
    auto model = examples::mnist_mlp();
    dl::io::save(output / "initial.ct", model->state_dict());
    const dl::StateDict sampling{{"indices", indices}};
    dl::io::save(output / "order.ct", sampling);
    dl::optim::SGD optimizer(model->parameters(), {.learning_rate = 0.1});
    auto stream = [&](const char* name) {
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(output / name);
        file << std::setprecision(12);
        return file;
    };
    auto curve = stream("ctorch-curve.csv");
    curve << "step,loss\n";
    for (dl::Index step = 0; step < steps; ++step) {
        auto selected = indices.select(0, step);
        auto input = dl::index_select(data.input, 0, selected);
        auto target = dl::index_select(data.target, 0, selected);
        optimizer.zero_grad();
        auto loss = dl::cross_entropy((*model)(input), target);
        loss.backward();
        optimizer.step();
        curve << step + 1 << ',' << loss.item<float>() << '\n';
    }
    dl::io::save(output / "ctorch-final.ct", model->state_dict());
    auto test = dl::data::load_mnist(directory / "t10k-images-idx3-ubyte", directory / "t10k-labels-idx1-ubyte");
    model->eval();
    dl::NoGradGuard no_grad;
    double total_loss = 0;
    dl::Index correct = 0;
    for (dl::Index start = 0; start < test.size(); start += 256) {
        const auto count = std::min(dl::Index{256}, test.size() - start);
        auto logits = (*model)(test.input.narrow(0, start, count));
        auto labels = test.target.narrow(0, start, count);
        total_loss += dl::cross_entropy(logits, labels, dl::LossReduction::sum).item<float>();
        correct += static_cast<dl::Index>(dl::sum(dl::eq(dl::argmax(logits, 1), labels).astype(dl::DType::f32)).item<float>());
    }
    auto report = stream("ctorch-training.json");
    report << "{\"steps\":" << steps << ",\"test_loss\":" << total_loss / test.size()
           << ",\"test_accuracy\":" << static_cast<double>(correct) / test.size() << "}\n";
    std::cout << "test loss " << total_loss / test.size() << ", accuracy " << static_cast<double>(correct) / test.size() << '\n';
}
