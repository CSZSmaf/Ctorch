#include "project_models.h"

#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) {
        std::cerr << "usage: ctorch_text_generate <weights.ct> <prefix>"
                  << " [--model small|medium] [--tokens N] [--temperature value] [--device cpu|index]\n";
        return 1;
    }
    auto config = examples::medium_text;
    auto device = dl::Device::cpu();
    dl::Index tokens = 200;
    double temperature = 0.8;
    for (int i = 3; i < argc; i += 2) {
        const std::string_view name = argv[i], value = argv[i + 1];
        if (name == "--model") config = value == "small" ? examples::TextModelOptions{} : examples::medium_text;
        else if (name == "--tokens") tokens = std::stoll(std::string(value));
        else if (name == "--temperature") temperature = std::stod(std::string(value));
        else if (name == "--device") device = value == "cpu" ? dl::Device::cpu() : dl::Device::cuda(std::stoi(std::string(value)));
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    examples::ByteTransformer model(config);
    model.load_state_dict(dl::io::load(argv[1]));
    model.to(device);
    model.eval();
    dl::NoGradGuard no_grad;
    std::mt19937 random(42);
    std::string text = argv[2];
    std::cout << text;
    for (dl::Index step = 0; step < tokens; ++step) {
        const auto length = std::min(config.length, static_cast<dl::Index>(text.size()));
        std::vector<dl::Index> bytes(text.end() - length, text.end());
        for (auto& byte : bytes) byte = static_cast<unsigned char>(byte);
        auto input = dl::from_span<dl::Index>(bytes, {1, length}, false, device);
        auto logits = model(input).select(1, length - 1).reshape({256});
        auto probabilities = dl::softmax(logits / temperature, 0).to_vector<float>();
        const auto next = static_cast<char>(std::discrete_distribution<int>(probabilities.begin(), probabilities.end())(random));
        text += next;
        std::cout << next << std::flush;
    }
    std::cout << '\n';
}
