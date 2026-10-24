#include "../project_models.h"
#include "../cifar_data.h"
#include "../text_data.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

namespace {

std::unique_ptr<dl::nn::UnaryModule> make_model(std::string_view name) {
    if (name == "mlp") return examples::mnist_mlp();
    if (name == "lenet") return examples::lenet();
    if (name == "resnet") return examples::resnet20();
    if (name == "transformer") return std::make_unique<examples::ByteTransformer>(examples::medium_text);
    throw std::invalid_argument("unknown model: " + std::string(name));
}

dl::data::Batch inputs(std::string_view model, const std::filesystem::path& path) {
    if (model == "transformer") {
        std::ifstream source;
        source.exceptions(std::ios::failbit | std::ios::badbit);
        source.open(path, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
        const auto test = std::string_view(text).substr(text.size() * 95 / 100);
        const examples::TextDataset dataset{test, examples::medium_text.length};
        const dl::Index selected[] = {0, 1};
        return dataset(selected);
    }
    auto dataset = model == "resnet" ? examples::load_cifar10(path, false)
        : dl::data::load_mnist(path / "t10k-images-idx3-ubyte", path / "t10k-labels-idx1-ubyte");
    const dl::Index count = model == "resnet" ? 4 : model == "lenet" ? 16 : 32;
    return {dataset.input.narrow(0, 0, count), dataset.target.narrow(0, 0, count)};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: ctorch_project_compare <mlp|lenet|resnet|transformer>"
                  << " <weights.ct> <dataset-path> <output-directory> [cpu|device-index]\n";
        return 1;
    }
    const std::string name = argv[1];
    const auto device = argc == 6 && std::string_view(argv[5]) != "cpu"
        ? dl::Device::cuda(std::stoi(argv[5])) : dl::Device::cpu();
    const std::filesystem::path output = argv[4];
    std::filesystem::create_directories(output);
    auto model = make_model(name);
    model->load_state_dict(dl::io::load(argv[2]));
    model->to(device);
    // eval 固定 Dropout 和 BatchNorm 的行为，求导仍然开启。
    model->eval();
    auto batch = inputs(name, argv[3]);
    batch.input = batch.input.to(device);
    batch.target = batch.target.to(device);
    dl::io::save(output / "weights.ct", model->state_dict());
    auto logits = (*model)(batch.input);
    auto loss = dl::cross_entropy(logits.reshape({batch.target.numel(), logits.shape().back()}),
                                  batch.target.reshape({-1}), dl::LossReduction::mean, -100);
    loss.backward();
    dl::StateDict records{{"input", batch.input}, {"target", batch.target}, {"logits", logits}, {"loss", loss}};
    dl::Index parameters = 0;
    for (const auto& [key, parameter] : model->named_parameters()) {
        parameters += parameter.numel();
        records.emplace_back("grad." + key, *parameter.grad());
    }
    dl::optim::SGD optimizer(model->parameters(), {.learning_rate = 0.01});
    optimizer.step();
    for (const auto& [key, parameter] : model->named_parameters()) records.emplace_back("updated." + key, parameter);
    dl::io::save(output / "results.ct", records);

    // 预热后对七组连续前向计时，每组五次；设备同步包含真实计算完成时间。
    dl::NoGradGuard no_grad;
    for (int i = 0; i < 3; ++i) (*model)(batch.input);
    dl::synchronize(device);
    std::vector<double> samples;
    for (int group = 0; group < 7; ++group) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) (*model)(batch.input);
        dl::synchronize(device);
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5);
    }
    std::sort(samples.begin(), samples.end());
    std::ofstream report;
    report.exceptions(std::ios::failbit | std::ios::badbit);
    report.open(output / "ctorch.json");
    report << std::setprecision(10) << "{\n  \"model\": \"" << name << "\",\n  \"parameters\": " << parameters
           << ",\n  \"loss\": " << loss.item<float>() << ",\n  \"forward_ms\": " << samples[3]
           << ",\n  \"device\": \"" << (device.type == dl::DeviceType::cpu ? "cpu" : "cuda") << "\"\n}\n";
    std::cout << name << ": " << parameters << " parameters, loss " << loss.item<float>()
              << ", median forward " << samples[3] << " ms\n";
}
