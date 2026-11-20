#include "project_models.h"
#include <dl/onnx.h>

#include <iostream>
#include <map>

int main(int argc, char** argv) try {
    if (argc < 2 || argc > 4) {
        std::cout << "usage: ctorch_export_onnx <model.onnx> [mnist-best.ct [batch-size]]\n";
        return argc < 2 ? 0 : 1;
    }
    dl::manual_seed(12);
    const bool mnist = argc >= 3;
    const dl::Index batch = argc == 4 ? std::stoll(argv[3]) : (mnist ? 1 : 2);
    if (batch <= 0) throw std::invalid_argument("batch size must be positive");
    auto model = mnist ? examples::mnist_mlp() : examples::two_layer_mlp(4, 8, 3);
    if (mnist) model->load_state_dict(dl::io::load(argv[2]));

    dl::plan::Graph graph;
    auto x = graph.input(dl::contiguous_spec(mnist ? dl::Shape{batch, 1, 28, 28} : dl::Shape{batch, 4}), "input");
    if (mnist) x = graph.call(dl::plan::ops::reshape({batch, 784}), {x});

    std::map<std::string, dl::plan::Value> parameter;
    std::vector<dl::onnx::Binding> weights;
    for (const auto& [name, tensor] : model->named_parameters()) {
        auto value = graph.input(dl::tensor_spec(tensor), name);
        parameter.emplace(name, value);
        weights.push_back({value, tensor});
    }
    auto y = graph.call(dl::plan::ops::linear(), {x, parameter.at("hidden.weight"), parameter.at("hidden.bias")});
    y = graph.call(dl::plan::ops::relu(), {y});
    y = graph.call(dl::plan::ops::linear(), {y, parameter.at("output.weight"), parameter.at("output.bias")});
    if (!mnist) y = graph.call(dl::plan::ops::softmax(-1), {y});
    dl::onnx::save(argv[1], graph, std::span(&y, 1), weights);
    if (mnist) std::cout << "input: float32[" << batch << ",1,28,28], output_0: float32[" << batch << ",10]\n";
    else std::cout << "input: float32[2,4], output_0: float32[2,3]\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
