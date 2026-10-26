#include <dl/dl.h>
#include <dl/onnx.h>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "usage: ctorch_export_onnx <model.onnx>\n"; return 0; }
    dl::manual_seed(12);
    dl::nn::Linear hidden(4, 8), output(8, 3);
    dl::plan::Graph graph;
    auto x = graph.input(dl::contiguous_spec({2, 4}), "input");
    std::vector<dl::onnx::Binding> weights;
    auto bind = [&](const dl::Tensor& tensor, std::string name) {
        auto value = graph.input(dl::tensor_spec(tensor), std::move(name));
        weights.push_back({value, tensor});
        return value;
    };
    auto w1 = bind(hidden.weight(), "hidden.weight"), b1 = bind(*hidden.bias(), "hidden.bias");
    auto w2 = bind(output.weight(), "output.weight"), b2 = bind(*output.bias(), "output.bias");
    auto y = graph.call(dl::plan::ops::linear(), {x, w1, b1});
    y = graph.call(dl::plan::ops::relu(), {y});
    y = graph.call(dl::plan::ops::linear(), {y, w2, b2});
    y = graph.call(dl::plan::ops::softmax(-1), {y});
    dl::onnx::save(argv[1], graph, std::span(&y, 1), weights);
    std::cout << "input: float32[2,4], output_0: float32[2,3]\n";
}
