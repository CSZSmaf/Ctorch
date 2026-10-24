#include "dl/io.h"
#include "dl/npu.h"

#include <iostream>
#include <string>
#include <utility>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: ctorch_npu_infer model.onnx inputs.ct outputs.ct\n";
        return 1;
    }

    // 输入文件使用 io::save 的张量格式，条目顺序对应模型的输入端口。
    dl::npu::CompiledModel model(argv[1]);
    auto inputs = dl::io::load(argv[2]);
    auto request = model.create_request();
    for (std::size_t i = 0; i < model.inputs().size(); ++i) request.set_input(i, inputs[i].second);

    // 输出缓冲按固定端口分配一次，连续推理时沿用同一状态表与存储。
    dl::StateDict outputs;
    outputs.reserve(model.outputs().size());
    for (std::size_t i = 0; i < model.outputs().size(); ++i) {
        const auto& port = model.outputs()[i];
        auto name = port.name.empty() ? "output_" + std::to_string(i) : port.name;
        outputs.push_back({std::move(name), dl::zeros(port.shape, {.dtype = port.dtype})});
    }

    request.start_async();
    request.wait();
    for (std::size_t i = 0; i < outputs.size(); ++i) request.output_into(i, outputs[i].second);
    dl::io::save(argv[3], outputs);
    std::cout << "Saved " << outputs.size() << " output tensors\n";
}
